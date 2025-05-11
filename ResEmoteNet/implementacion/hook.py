# Un hook en PyTorch es una función que se puede "enganchar" (hook) a una capa o módulo del modelo.
# Permite capturar lo que entra, sale o fluye como gradiente por esa capa durante el forward o backward.
# Es útil para depurar, visualizar activaciones, gradientes o entender mejor cómo funciona el modelo internamente.

class Hook():
    def __init__(self):
        # Estos van a almacenar los objetos hook que se registran
        self.hook_forward = None
        self.hook_backward = None

        # Aquí se guardan los valores que salieron en el forward y backward
        self.forward_out = None
        self.backward_out = None

    def hook_fn_forward(self, module, input, output):
        # Esta función guarda el output del módulo cuando se hace forward
        self.forward_out = output

    def hook_fn_backward(self, module, grad_input, grad_output):
        # Esta guarda el gradiente de salida durante el backward
        self.backward_out = grad_output[0]  # grad_output es una tupla

    def register_hook(self, module):
        # Enganchamos las funciones hook al módulo: forward y backward
        self.hook_forward = module.register_forward_hook(self.hook_fn_forward)
        self.hook_backward = module.register_full_backward_hook(self.hook_fn_backward)

    def unregister_hook(self):
        # Quitamos los hooks cuando ya no los necesitamos
        self.hook_forward.remove()
        self.hook_backward.remove()
