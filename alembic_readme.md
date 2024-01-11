## **Alembic migrations:**

`alembic init alembic` # initialize the alembic

`alembic revision --autogenerate -m "Create user model"` # first migration

`alembic upgrade 66b63a` # reflect migration on database (here 66b63a) is ssh value