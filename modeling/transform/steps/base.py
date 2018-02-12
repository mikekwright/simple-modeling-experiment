from abc import abstractmethod, ABC
from typing import Dict, Sequence, Any, Tuple


class BaseStep(ABC):
    def __init__(self, field_names: Sequence[str], result_names: Sequence[str] = None):
        self._field_names = field_names
        self._result_names = result_names

    @property
    def field_names(self) -> str:
        return self._field_names

    @property
    def result_names(self) -> str:
        return self._result_names if self._result_names else self.field_names

    def two_pass(self) -> bool:
        return False

    def prep_pass(self, train_data: Sequence[Tuple[Dict, Any]]):
        pass

    def __call__(self, point: Dict, meta: Dict) -> Dict:
        for field, result in zip(self.field_names, self.result_names):
            if field not in point and field not in meta:
                raise ValueError(f'Unknown field {field} requested')

            value = meta.get(field, point[field])
            response = self._handle_call(field, value)
            meta[result] = response

        return meta

    @abstractmethod
    def _handle_call(self, field: str, value: Any) -> Any:
        pass

    @abstractmethod
    def store_results(self, directory: str):
        pass
