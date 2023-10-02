from injector import Injector


def create_application_injector() -> Injector:
    injector = Injector(auto_bind=True)
    return injector


root_injector: Injector = create_application_injector()
