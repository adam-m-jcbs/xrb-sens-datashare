from inspect import signature
from types import MethodType

# same names are used to install hooks in KEPLER process (server)

# alias names where things are identical to be transoparent to user

class CycleHooks(object):
    def _hook_cycle(self, cycler = None):
        if cycler is None:
            cycler = self._cycle
        if self._debug:
            print(f' [CYCLE] executing cycle {self.qparm.ncyc}')
        for h in self._pre_cycle_hooks:
            if self._debug:
                print(f' [CYCLE] executing pre-cycle {h}')
            h()
        result = cycler()
        for h in self._post_cycle_hooks:
            if self._debug:
                print(f' [CYCLE] executing post-cycle {h}')
            h()
        return result

    def _server_add_hook(self, hook, which = 'post', append = None):
        if which == 'pre':
            hooks = self._pre_cycle_hooks
            append_default = True
        else:
            hooks = self._post_cycle_hooks
            append_default = False
        if append is None:
            append = append_default
        if append:
            index = len(hooks)
        else:
            index = 0
        # actual free parameters
        parm = signature(hook).parameters
        if len(parm) == 1:
            hook = MethodType(hook, self)
        elif len(parm) == 0:
            pass
        else:
            raise AssertionError(f' [ADD_HOOK] requires call signature Hook() or hook(self): {hook}: {spec}')
        if hook in hooks:
            raise Exception(f' [ADD_HOOK] hook {hook} has already been added.')
        hooks.insert(index, hook)

    def server_add_pre_hook(self, hook, append = False):
        self._server_add_hook(hook, which = 'pre', append = append)

    def server_add_post_hook(self, hook, append = True):
        self._server_add_hook(hook, which = 'post', append = append)

    def server_clear_pre_hooks(self):
        self._pre_cycle_hooks = []

    def server_clear_post_hooks(self):
        self._post_cycle_hooks = []

    def server_clear_hooks(self):
        self.server_clear_pre_hooks()
        self.server_clear_post_hooks()

    add_pre_hook     = client_add_pre_hook     = server_add_pre_hook
    add_post_hook    = client_add_post_hook    = server_add_post_hook
    clear_pre_hooks  = client_clear_pre_hooks  = server_clear_pre_hooks
    clear_post_hooks = client_clear_post_hooks = server_clear_post_hooks
    clear_hooks      = client_clear_hooks      = server_clear_hooks
