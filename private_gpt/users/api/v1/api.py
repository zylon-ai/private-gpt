from private_gpt.users.api import deps
from private_gpt.users.api.v1.routers import auth, roles, user_roles, users, subscriptions, companies, departments, documents, audits, chat_history
from fastapi import APIRouter

api_router = APIRouter(prefix="/v1")

api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(roles.router)
api_router.include_router(user_roles.router)
api_router.include_router(companies.router)
api_router.include_router(subscriptions.router)
api_router.include_router(departments.router)
api_router.include_router(documents.router)
api_router.include_router(audits.router)
api_router.include_router(chat_history.router)

