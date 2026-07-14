import asyncio
import threading
from typing import Any

from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.di import (
    clean_global_injector,
    get_global_injector,
    get_injector,
)


def test_injector_is_shared_across_loops_when_global_root_exists() -> None:
    class IdentifiableService:
        instance_count = 0

        def __init__(self) -> None:
            self.id: int = id(self)
            self.instance_number: int = IdentifiableService.instance_count
            IdentifiableService.instance_count += 1
            self.created_in_thread: int = threading.get_ident()

        async def operation(self, should_fail: bool = False) -> dict[str, Any]:
            if should_fail:
                raise ValueError("Operation failed intentionally")
            return {
                "service_id": self.id,
                "instance_number": self.instance_number,
                "thread_id": threading.get_ident(),
            }

    rollback_called: bool = False
    error_value: Exception | None = None

    def rollback_fn(error: Exception) -> None:
        nonlocal rollback_called, error_value
        rollback_called = True
        error_value = error

    def run_task_in_thread(task_id: int, should_fail: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_id": task_id,
            "thread_id": None,
            "service_id": None,
            "instance_number": None,
            "error": None,
            "success": False,
        }

        thread_injector = None

        def thread_func() -> None:
            nonlocal thread_injector
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:

                async def async_task() -> None:
                    nonlocal thread_injector
                    injector = get_injector()
                    thread_injector = injector

                    service = IdentifiableService()
                    injector.binder.bind(IdentifiableService, to=service)

                    try:
                        retrieved_service = injector.get(IdentifiableService)
                        result["service_id"] = retrieved_service.id
                        result["instance_number"] = retrieved_service.instance_number
                        result["thread_id"] = threading.get_ident()

                        service_result = await retrieved_service.operation(should_fail)
                        result.update(service_result)
                        result["success"] = True
                    except Exception as e:
                        result["error"] = e
                        rollback_fn(e)

                loop.run_until_complete(async_task())
            finally:
                loop.close()

        thread = threading.Thread(target=thread_func)
        thread.start()
        thread.join(timeout=5.0)

        result["injector"] = thread_injector

        return result

    results: list[dict[str, Any]] = []
    for i in range(3):
        should_fail = i % 3 == 2
        result = run_task_in_thread(i, should_fail)
        results.append(result)

    injectors = [r["injector"] for r in results]
    injector_ids = [id(inj) for inj in injectors]
    assert len(set(injector_ids)) == 1

    service_ids = [r["service_id"] for r in results]
    assert len(set(service_ids)) == len(results)

    assert rollback_called
    assert isinstance(error_value, ValueError)

    for i, result in enumerate(results):
        should_fail = i % 3 == 2
        if should_fail:
            assert result["error"] is not None
            assert not result["success"]
        else:
            assert result["error"] is None
            assert result["success"]

    print("\nDetailed test results:")
    for i, r in enumerate(results):
        print(f"\nTask {i}:")
        print(f"  Thread ID: {r['thread_id']}")
        print(f"  Injector ID: {id(r['injector'])}")
        print(f"  Service ID: {r['service_id']}")
        print(f"  Service Instance #: {r['instance_number']}")
        print(f"  Success: {r['success']}")
        if r["error"]:
            print(f"  Error: {r['error']}")


def test_global_fallback_injector() -> None:
    class MarkerService:
        def __init__(self) -> None:
            self.id: int = id(self)

    global_injector = get_global_injector()
    marker = MarkerService()
    global_injector.binder.bind(MarkerService, to=marker)

    retrieved_injector = get_global_injector()
    retrieved_marker = retrieved_injector.get(MarkerService)

    assert id(global_injector) == id(retrieved_injector)
    assert retrieved_marker.id == marker.id

    standard_injector = get_injector()
    assert id(standard_injector) == id(global_injector)


def test_clean_global_injector() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run_in_loop() -> None:
        old_global_injector = get_global_injector()
        assert old_global_injector is not None
        old_global_injector.get(VectorStoreComponent)
        old_global_injector.get(NodeStoreComponent)

        running_loop = asyncio.get_running_loop()
        await clean_global_injector(running_loop)

        new_injector = get_global_injector()
        assert new_injector is not None
        assert new_injector == old_global_injector

    loop.run_until_complete(run_in_loop())
