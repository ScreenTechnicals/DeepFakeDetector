from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    model: str
    probability: float = Field(ge=0.0, le=1.0)
    prediction: int = Field(ge=0, le=1)
    class_name: str = Field(serialization_alias="class")
    inference_time: float

    def model_dump(self, *args, **kwargs):
        by_alias = kwargs.pop("by_alias", False)

        if hasattr(BaseModel, "model_dump"):
            data = BaseModel.model_dump(
                self, *args, by_alias=by_alias, **kwargs
            )
        else:
            data = self.dict(*args, by_alias=by_alias, **kwargs)

        if by_alias and "class_name" in data and "class" not in data:
            data["class"] = data.pop("class_name")

        return data
