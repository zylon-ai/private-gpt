from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    declared_attr,
    mapped_column,
    relationship,
)


class SkillBase(DeclarativeBase):
    pass


class SkillORM(SkillBase):
    __tablename__ = "skills"

    @declared_attr.directive
    def __table_args__(cls) -> dict[str, str]:
        return {"schema": "app"}

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    collection: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    display_title: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    loading: Mapped[str] = mapped_column(String(16), nullable=False)
    readonly: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(tz=UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(tz=UTC),
        onupdate=lambda: datetime.now(tz=UTC),
    )

    versions: Mapped[list["SkillVersionORM"]] = relationship(
        "SkillVersionORM",
        back_populates="skill",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class SkillVersionORM(SkillBase):
    __tablename__ = "skill_versions"

    @declared_attr.directive
    def __table_args__(cls) -> tuple[UniqueConstraint, dict[str, str]]:
        return (
            UniqueConstraint("skill_id", "version", name="uq_skill_version"),
            {"schema": "app"},
        )

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    skill_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("app.skills.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    version: Mapped[str] = mapped_column(String(255), nullable=False)
    frontmatter_json: Mapped[dict[str, object]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )
    storage_prefix: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(tz=UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(tz=UTC),
        onupdate=lambda: datetime.now(tz=UTC),
    )

    skill: Mapped[SkillORM] = relationship("SkillORM", back_populates="versions")
