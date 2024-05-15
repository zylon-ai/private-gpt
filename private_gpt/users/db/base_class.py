from typing import Any

import inflect  
from sqlalchemy.ext.declarative import as_declarative, declared_attr

p = inflect.engine() 

@as_declarative()
class Base:
    id: Any
    __name__: str
    # Generate __tablename__ automatically
    @declared_attr
    def __tablename__(cls) -> str:
        return p.plural(cls.__name__.lower())