from .api_resource import APIResource
from ..types import (
    ScoreType,
    ScoringFunction,
    CustomScoringConfig,
    InputType,
    OutputType,
    Score,
)

class RegisteredScoringFunction(ScoringFunction):
    registration: CustomScoringConfig

class ScoringHelper(APIResource):
    _initialized: bool = False
    _registered_functions: list[RegisteredScoringFunction]
    _standard_scoring: list[ScoreType]
    _custom_scoring: list[ScoringFunction]

    def __init__(self, client, scoring: list[ScoreType]):
        super().__init__(client)
        self._standard_scoring = [s for s in scoring if type(s) == ScoreType]
        self._custom_scoring = [s for s in scoring if type(s) == ScoringFunction]

    def initialize(self):
        if self._initialized:
            return
        self._register_scoring_functions()
        self._initialized = True

    def get_config(self) -> list[ScoreType | CustomScoringConfig]:
        if not self._initialized:
            raise Exception("ScoringHelper is not initialized")
        return self._standard_scoring + [f.registration for f in self._registered_functions]
    
    def score(self, input: InputType, expected: OutputType, actual: OutputType) -> dict[str, Score]:
        if not self._initialized:
            raise Exception("ScoringHelper is not initialized")
        scores = {}
        for f in self._registered_functions:
            score = f.scorer.score_fn({ 
                "input": input, 
                "expected": expected, 
                "output": actual 
            })
            scores[f.registration.key_name] = score
        return scores
    
    def _register_scoring_functions(self):
        scoring_obj = [{
            "name": f.name,
            "version": f.version,
            "score_config": f.score_config.model_dump(),
            "execution_config": {},
        } for f in self._custom_scoring]

        data = self._client.request(
            "POST",
            "/scoring/register-functions",
            json={
                "scoring": scoring_obj
            }
        )
        registrations = data["scoring"]
        self._registered_functions = [
            RegisteredScoringFunction(
                **f.model_dump(),
                registration=registrations[idx]
            ) 
        for idx, f in enumerate(self._custom_scoring)]
