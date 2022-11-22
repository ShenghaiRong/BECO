import inspect


class Register:
    def __init__(self, name:str) -> None:
        self.name = name
        self._module_dict = dict()

    def get(self, key: str):
        """
        If a Register has chilren Registers, either XXX.XXX.get(mod_name) or
        XXX.get(XXX.mod_name) can be used
        """
        scope, real_key = self._split_scope(key)
        if scope is None:
            assert real_key in self._module_dict, f"{real_key} does not exists"
            return self._module_dict[real_key]
        else:
            assert hasattr(self, scope)
            assert isinstance(getattr(self, scope), Register)
            return getattr(self, scope).get(real_key)

    def _get_scope(self):
        # inspect.stack() trace where this function is called, the index-2
        # indicates the frame where `infer_scope()` is called
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split('.')
        return split_filename[0]

    def _split_scope(self, key: str):
        split_index = key.find('.')
        if split_index != -1:
            return key[:split_index], key[split_index + 1:]
        else:
            return None, key

    def add_children(self, register):
        assert not hasattr(self, register.name)
        setattr(self, register.name, register)

    def _register_module(self, name:str=None, module=None, force:bool=False):
        if name is None:
            name = module.__name__
        if not force and name in self._module_dict:
            raise KeyError(f'{name} is already registered in {self.name}')
        self._module_dict[name] = module

    def register_module(self, name:str=None, module=None, force:bool=False):
        # NOTE: This is a dispatcher design
        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(name, module, force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(name, cls, force)
            return cls
        return _register

