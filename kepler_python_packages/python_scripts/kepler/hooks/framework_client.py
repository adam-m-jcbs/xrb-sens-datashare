# this had to be separated out due to dependence on kepler.process
# module as it therefore cannot be loaded with framework for kepler
# main module hooks.

# overwrite hooks to "server"; default will be "server"

from .framework import *

from kepler.process.api import Call

class ClientCycleHooks(CycleHooks):
    def server_add_pre_hook(self, hook, append = False):
        task = Call(
            'add_pre_hook',
            args = (hook,),
            kwargs = dict(append = append),
            )
        self.servercall(task)

    def server_add_post_hook(self, hook, append = False):
        task = Call(
            'add_post_hook',
            args = (hook,),
            kwargs = dict(append = append),
            )
        self.servercall(task)

    def server_clear_pre_hooks(self):
        task = Call(
            'clear_pre_hooks',
            )
        self.servercall(task)

    def server_clear_post_hooks(self):
        task = Call(
            'clear_post_hooks',
            )
        self.servercall(task)

    def server_clear_hooks(self):
        task = Call(
            'clear_hooks',
            )
        self.servercall(task)

    add_pre_hook     = CycleHooks.client_add_pre_hook
    add_post_hook    = CycleHooks.client_add_post_hook
    clear_pre_hooks  = CycleHooks.client_clear_pre_hooks
    clear_post_hooks = CycleHooks.client_clear_post_hooks
    clear_hooks      = CycleHooks.client_clear_hooks
