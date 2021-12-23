class RuntimeError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)


class NotImplementedError(RuntimeError):
    def __init__(self):
        super().__init__("Functionality not implemented.")


class InvalidActivationFunctionError(RuntimeError):
    def __init__(self, activation_name: str):
        super().__init__("{} activation function is not supported.".format(activation_name))
