from injector import Injector


def create_application_injector() -> Injector:
    # For the future, configure dependencies here
    return Injector(auto_bind=True)


root_injector: Injector = create_application_injector()
