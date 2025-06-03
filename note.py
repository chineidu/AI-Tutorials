from enum import Enum
from typing import Annotated, Any

from pydantic import (  # type: ignore
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
)
from pydantic.alias_generators import to_camel


def round_probability(value: float) -> float:
    """Round a float value to two decimal places.

    Returns:
        float: Rounded value.
    """
    if isinstance(value, float):
        return round(value, 2)
    return value


class BaseSchema(BaseModel):
    """Base schema class that inherits from Pydantic BaseModel.

    This class provides common configuration for all schema classes including
    camelCase alias generation, population by field name, and attribute mapping.
    """

    model_config: ConfigDict = ConfigDict(  # type: ignore
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )


Float = Annotated[float, BeforeValidator(round_probability)]
String = Annotated[
    str, StringConstraints(strip_whitespace=True, strict=True, min_length=2, max_length=50)
]


class EntityType(str, Enum):
    TELECOM_SERVICES = "telecomServices"
    LEVIES_AND_CHARGES = "leviesAndCharges"
    CABLE_TV_OR_STREAMING_OR_SUBSCRIPTIONS = "cableTvOrStreamingOrSubscriptions"
    UTILITIES = "utilities"
    ENERGY_AND_FUEL = "energyAndFuel"
    LEISURE_LIFESTYLE_AND_RECREATION = "leisureLifestyleAndRecreation"
    HEALTH_ACTIVITY = "healthActivity"
    LOAN_LENDER = "loanLender"
    SAVINGS_AND_INVESTMENTS = "savingsAndInvestments"
    BETTING_AND_GAMBLING = "bettingAndGambling"
    LOCATION = "location"
    PERSON = "person"
    TRANSACTION_REASON = "transactionReason"
    RELIGIOUS_ACTIVITY = "religiousActivity"
    MISC = "miscellaneous"


class Entity(BaseSchema):
    """A schema for representing named entity recognition output."""

    text: String = Field(description="Text of the entity")
    label: EntityType = Field(description="Label of the entity")  # type: ignore
    score: Float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score of the entity. Range: 0.0 (least confident) "
        "to 1.0 (most confident)",
    )


class EntitySchemaResponse(BaseSchema):
    """A schema for representing a collection of NER responses with metadata."""

    id: str = Field(description="ID of the input data", alias="txnId")
    text: str | None = Field(default=None, description="The original input data")
    entities: list[Entity | list] = Field(default_factory=list, description="A list of entities")
    reasoning: str | None = Field(
        default=None,
        description="The reasoning behind the entity extraction. This is a string "
        "representation of the reasoning.",
    )

    def to_sqlalchemy_dict(self) -> dict[str, Any]:
        """Convert to dictionary with SQLAlchemy attribute names."""
        return {
            "txn_id": self.id,
            "text": self.text,
            "entities": [
                entity.model_dump() if hasattr(entity, "model_dump") else entity
                for entity in self.entities
            ],
        }


class AllEntitySchemaResponse(BaseSchema):
    """A schema for representing a collection of entity schema responses."""

    data: list[EntitySchemaResponse]
