import json
import logging
import sys

from inspect import signature


logger = logging.getLogger(__name__)


class JsonLoader:
    """
    The json initializer will take a json configuration and initialize the underlying types
    """
    def __init__(self, class_lookup_func=None, inject_dict=None, refs=None):
        logger.debug('New instance of json initializer created')
        self.refs = refs
        if self.refs is None:
            self.refs = {}
        self.class_lookup_func = class_lookup_func or get_class_from_name
        self.inject_dict = inject_dict or {}

    def initialize_json(self, json_config):
        logger.debug(f'Request to initialize from json {json_config}')
        raw_config = json.loads(json_config)
        if '__refs__' in raw_config:
            self._load_refs(raw_config['__refs__'])
            del raw_config['__refs__']

        adjusted_config = json.dumps(raw_config)
        return json.loads(adjusted_config, object_hook=self._json_object_hook)

    def _load_refs(self, ref_config):
        ref_config = json.dumps(ref_config)
        json.loads(ref_config, object_hook=self._json_object_hook)

    def _augment_config_with_injects(self, class_, dct):
        class_params = signature(class_).parameters

        for key, value in self.inject_dict.items():
            if key in dct:
                continue

            if key in class_params:
                logger.debug(f'Injecting {key} into {class_}')
                dct[key] = value

        return dct

    def _json_object_hook(self, dct):
        ref_name = None
        if '__ref_name__' in dct:
            ref_name = dct['__ref_name__']
            dct.pop('__ref_name__')
        if '__comment__' in dct:
            dct.pop('__comment__', None)
        if '__refs__' in dct:
            dct.pop('__refs__', None)
        if '__ref__' in dct:
            return self.refs[dct['__ref__']]
        if '__type__' in dct:
            class_ = self.class_lookup_func(dct['__type__'])
            dct.pop('__type__', None)
            try:
                dct = self._augment_config_with_injects(class_, dct)
                obj = class_(**dct)
            except Exception as ex:
                raise Exception(f'Failed to create object with class {class_}. JSON dict: {dct}') from ex
            if ref_name:
                self.refs[ref_name] = obj
            return obj
        return dct