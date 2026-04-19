"""Bootstrap demo user and chat session when DEMO_API_KEY is enabled."""

from app.core.config import settings
from app.core.logging import logger
from app.models.user import User
from app.services.database import DatabaseService


async def ensure_demo_identity(db: DatabaseService) -> None:
    """Create demo user + session if DEMO_API_KEY is set and rows are missing."""
    if not settings.DEMO_API_KEY:
        return

    user = await db.get_user_by_email(settings.DEMO_USER_EMAIL)
    if user is None:
        # Password is never used on the demo path (API key only)
        placeholder = "Demo#Pass9x!"
        user = await db.create_user(settings.DEMO_USER_EMAIL, User.hash_password(placeholder))
        logger.info("demo_user_created", email=settings.DEMO_USER_EMAIL, user_id=user.id)
    else:
        logger.info("demo_user_exists", email=settings.DEMO_USER_EMAIL, user_id=user.id)

    sess = await db.get_session(settings.DEMO_SESSION_ID)
    if sess is None:
        await db.create_session(settings.DEMO_SESSION_ID, user.id, name="Demo")
        logger.info("demo_session_created", session_id=settings.DEMO_SESSION_ID)
    else:
        logger.info("demo_session_exists", session_id=settings.DEMO_SESSION_ID)
