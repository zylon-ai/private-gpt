from private_gpt.users.core.config import settings

SQLALCHEMY_DATABASE_URI = "postgresql+psycopg2://{username}:{password}@{host}:{port}/{db_name}".format(
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    db_name=settings.DB_NAME,
    username=settings.DB_USER,
    password=settings.DB_PASSWORD,
)