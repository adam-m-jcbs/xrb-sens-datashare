import os
import os.path
import numpy as np

from cmd import Cmd
from textwrap import fill

class KeplerCmdLoop0(Cmd):
    def __init__(self, kepler):
        self.kepler = kepler
        self.prompt = kepler.kd.nameprob + '> '
        self.intro = """Use 'x' command or <^D> to exit."""
        super().__init__()

        self.histfile = os.path.join(os.path.expanduser("~"), ".kepler_cmd_history")

    def default(self, cmdline):
        # filter "s" and "g" commands
        cmd = cmdline.strip().split(' ')
        ret = None
        if cmdline.strip() == 's':
            self.kepler.s()
        elif cmdline.strip() in ('x', 'X', 'EOF', ):
            print()
            ret = True
        elif cmd[0] == 's':
            try:
                n = int(cmd[1])
                self.kepler.cycle(n)
            except:
                pass
        elif cmdline.strip() == 'g':
            self.kepler.g()
        elif cmdline.strip().startswith('!'):
            exec('self.kepler.' + cmdline.strip()[1:])
        else:
            self.kepler.execute(cmdline)
        if self.kepler.status == 'terminated':
            return True
        return ret

    def preloop(self):
        import readline
        try:
            readline.read_history_file(self.histfile)
            self.old_len = readline.get_current_history_length()
        except FileNotFoundError:
            self.old_len = 0

    def postloop(self):
        import readline
        new_len = readline.get_current_history_length()
        readline.set_history_length(1000)
        try:
            open(self.histfile, 'rb').close()
        except FileNotFoundError:
            open(self.histfile, 'wb').close()
        readline.append_history_file(
            new_len - self.old_len,
            self.histfile)


class KeplerCmdLoop():
    def __init__(self, kepler):
        self.kepler = kepler
        self.prompt = kepler.kd.nameprob + '> '
        self.intro = """Use 'x' command or <^D> to exit."""
        self.completekey = 'tab'
        self.histfile = os.path.join(os.path.expanduser("~"), ".kepler_cmd_history")

    def cmdloop(self):
        import readline
        print(self.intro)
        try:
            readline.read_history_file(self.histfile)
            self.old_len = readline.get_current_history_length()
        except FileNotFoundError:
            self.old_len = 0
        self.old_completer = readline.get_completer()
        readline.set_completer(self.complete)
        readline.parse_and_bind(self.completekey+": complete")
        readline.set_completion_display_matches_hook(self.cdm)
        readline.set_completer_delims(' ,')
        cont = True
        while cont:
            try:
                cmdlines = input(self.prompt)
            except EOFError:
                break
            except KeyboardInterrupt:
                print(self.intro)
                continue
            cmdlines = cmdlines.split(',')
            for cmdline in cmdlines:
                # filter "s" and "g" commands
                cmd = cmdline.strip().split(' ')
                if cmdline.strip() == 's':
                    self.kepler.s()
                elif cmdline.strip() in ('x', 'X',):
                    print()
                    cont = False
                    break
                elif cmd[0] == 's':
                    try:
                        n = int(cmd[1])
                        self.kepler.cycle(n)
                    except:
                        pass
                elif cmdline.strip() == 'g':
                    self.kepler.g()
                elif cmdline.strip().startswith('!'):
                    exec('self.kepler.' + cmdline.strip()[1:])
                else:
                    self.kepler.execute(cmdline)
                if self.kepler.status == 'terminated':
                    cont = False
                    break
        readline.set_completer(self.old_completer)
        new_len = readline.get_current_history_length()
        readline.set_history_length(1000)
        try:
            open(self.histfile, 'rb').close()
        except FileNotFoundError:
            open(self.histfile, 'wb').close()
        readline.append_history_file(
            new_len - self.old_len,
            self.histfile)
        print()
        return

    def complete(self, text, state):
        # print(f'[{state},{text},{len(text)}]')
        if state == 0:
            import readline
            origline = readline.get_line_buffer()
            line = origline
            begidx = readline.get_begidx()
            endidx = readline.get_endidx()
            line = line[:endidx].rsplit(',')[-1]
            tokens = line.split()
            self.completions = None
            if len(tokens) > 0:
                if tokens[0] in ('p', 'q',):
                    if tokens[0] == 'p':
                        keys = list(self.kepler.kd.parm._data.keys())[1:]
                    else:
                        keys = list(self.kepler.kd.qparm._data.keys())[1:]

                    if len(tokens) == 2 and tokens[1] == text:
                        try:
                            ip = int(tokens[1])
                        except ValueError:
                            ip = 0
                        if ip == 0:
                            try:
                                xp = tokens[1]
                                xp = xp.replace('d','e')
                                xp = xp.replace('D','e')
                                xp = xp.replace('E','e')
                                xp = float(xp)
                            except ValueError:
                                xp = np.nan
                        else:
                            xp = np.nan
                        if ip > 0 and ip <= len(keys) + 1:
                            self.completions = [keys[ip - 1]]
                        elif not np.isnan(xp) and tokens[0] == 'p':
                            self.completions = [
                                k for k in keys
                                if np.isclose(self.kepler.kd.parm[k], xp)]
                        elif not np.isnan(xp) and tokens[0] == 'q':
                            self.completions = [
                                k for k in keys
                                if np.isclose(self.kepler.kd.qparm[k], xp)]
                        elif tokens[1].startswith('*') and not tokens[1].endswith('*'):
                            self.completions = [
                               k for k in keys
                               if k.find(tokens[1][1:]) >= 0]
                        elif tokens[1].startswith('*') and tokens[1].endswith('*'):
                            self.completions = [
                               k for k in keys
                               if k.find(tokens[1][1:-1]) >= 0]
                        elif not tokens[1].startswith('*') and tokens[1].endswith('*'):
                            self.completions = [
                               k for k in keys
                               if k.find(tokens[1][:-1]) == 0]
                        else:
                            self.completions = [
                                k for k in keys
                                if k.startswith(tokens[1])]
                    elif len(tokens) == 1:
                        self.completions = keys
                    elif len(tokens) == 2 and text == '':
                        if tokens[1] in keys:
                            if tokens[0] == 'p':
                                value = self.kepler.kd.parm[tokens[1]]
                            else:
                                value = self.kepler.kd.qparm[tokens[1]]
                            self.completions = [str(value)]

                else:
                    # case one other token
                    self.completions = []
                    self.completions += [
                        k for k in list(self.kepler.kd.parm._data.keys())[1:]
                        if k.startswith(tokens[0])]
                    self.completions += [
                        k for k in list(self.kepler.kd.qparm._data.keys())[1:]
                        if k.startswith(tokens[0])]
            else:
                # 0 tokens: list all or print help message
                self.completions = ['p', 'q', 'o',]
                self.completions += [
                    k for k in list(self.kepler.kd.parm._data.keys())[1:]]
                self.completions += [
                    k for k in list(self.kepler.kd.qparm._data.keys())[1:]]

        try:
            return self.completions[state]
        except (IndexError, AttributeError):
            pass

    def cdm(self, *args):
        import readline
        origline = readline.get_line_buffer()
        print()
        tabsize = max(len(s) for s in args[1]) + 2
        print(fill('\t'.join(args[1]),tabsize = tabsize))
        print(self.prompt + origline, end = '', flush = True)

# import matplotlib.pylab as plt
# from prompt_toolkit import prompt
# from prompt_toolkit import prompt_async
# import asyncio
# class KeplerCmdLoop():
#     def __init__(self, kepler):
#         self.kepler = kepler
#         self.prompt = kepler.kd.nameprob + '> '

#     async def my_coroutine(self):
#         while True:
#             cmdline = await prompt_async(self.prompt)#, patch_stdout=True)
#             # filter "s" and "g" commands
#             cmd = cmdline.strip().split(' ')
#             if cmdline.strip() == 's':
#                 self.kepler.s()
#                 return
#             if cmdline.strip() == 'x':
#                 return True
#             if cmd[0] == 's':
#                 try:
#                     n = int(cmd[1])
#                     self.kepler.cycle(n)
#                     return
#                 except:
#                     pass
#             if cmdline.strip() == 'g':
#                 self.kepler.g()
#                 return
#             self.kepler.execute(cmdline)
#             print('Done.')

#     async def my_draw(self):
#         while True:
#             # plt.pause(0.001)
#             await asyncio.sleep(.01)
#             # for p in self.kepler.kp.plots:
#             #     p.fig.canvas.draw_idle()

#     def cmdloop(self):
#         loop = asyncio.get_event_loop()
#         loop.run_until_complete(asyncio.gather(
#             self.my_coroutine(),
#             self.my_draw()))
#         # result = loop.run_until_complete(f)
#         # print('result: ', result)

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

import matplotlib.pylab as plt
from time import sleep, time

class KeplerCmdLoop3():
    def __init__(self, kepler):
        self.kepler = kepler
        self.prompt = kepler.kd.nameprob + '> '
        self.intro = """Use 'x' command or <^D> to exit."""

    def hook(self, ctx):
        while True:
            if ctx.input_is_ready():
                return
            sleep(0.1)
            t = time()
            # plt.pause(0.0001)
            print(time() - t)

    def cmdloop(self):
        session = PromptSession(
            message = self.prompt,
            history = FileHistory(os.path.expanduser('~/.kepler_history')),
            auto_suggest = AutoSuggestFromHistory(),
            inputhook = self.hook,
            )
        print(self.intro)
        cont = True
        while cont:
            try:
                cmdlines = session.prompt()
            except KeyboardInterrupt:
                print(self.intro)
                continue
            except EOFError:
                break
            cmdlines = cmdlines.split(',')
            for cmdline in cmdlines:
                # filter "s" and "g" commands
                cmd = cmdline.strip().split(' ')
                if cmdline.strip() == 's':
                    self.kepler.s()
                elif cmdline.strip() in ('x', 'X',):
                    print()
                    cont = False
                    break
                elif cmd[0] == 's':
                    try:
                        n = int(cmd[1])
                        self.kepler.cycle(n)
                    except:
                        pass
                elif cmdline.strip() == 'g':
                    self.kepler.g()
                elif cmdline.strip().startswith('!'):
                    exec('self.kepler.' + cmdline.strip()[1:])
                else:
                    self.kepler.execute(cmdline)
                if self.kepler.status == 'terminated':
                    cont = False
                    break
        return


from prompt_toolkit.eventloop import use_asyncio_event_loop
from prompt_toolkit.patch_stdout import patch_stdout

import asyncio
class KeplerCmdLoop3():
    def __init__(self, kepler):
        self.kepler = kepler
        self.prompt = kepler.kd.nameprob + '> '
        self.intro = """Use 'x' command or <^D> to exit."""
        self.cont = True

    async def my_coroutine(self):
        session = PromptSession(
            message = self.prompt,
            history = FileHistory(os.path.expanduser('~/.kepler_history')),
            auto_suggest = AutoSuggestFromHistory(),
            )
        print(self.intro)
        while self.cont:
            try:
                with patch_stdout():
                    cmdlines = await session.prompt(async_ = True)
            except KeyboardInterrupt:
                print(self.intro)
                continue
            except EOFError:
                self.cont = False
                break
            cmdlines = cmdlines.split(',')
            for cmdline in cmdlines:
                # filter "s" and "g" commands
                cmd = cmdline.strip().split(' ')
                if cmdline.strip() == 's':
                    self.kepler.s()
                elif cmdline.strip() in ('x', 'X',):
                    print()
                    self.cont = False
                    break
                elif cmd[0] == 's':
                    try:
                        n = int(cmd[1])
                        self.kepler.cycle(n)
                    except:
                        pass
                elif cmdline.strip() == 'g':
                    self.kepler.g()
                elif cmdline.strip().startswith('!'):
                    exec('self.kepler.' + cmdline.strip()[1:])
                else:
                    self.kepler.execute(cmdline)
                if self.kepler.status == 'terminated':
                    self.cont = False
                    break
        return

    async def my_draw(self):
        while self.cont:
            # plt.pause(0.01)
            await asyncio.sleep(0.1)
            # await asyncio.sleep(.01)
            # for p in self.kepler.kp.plots:
            #     p.fig.canvas.draw_idle()

    def cmdloop(self):
        loop = asyncio.get_event_loop()
        plt.interactive(True)
        #self.kepler.execute('plot 2')
        #import threading, multiprocessing
        #print(list(threading.enumerate()))
        #print(list(multiprocessing.enumerate()))
        #print(loop.is_running())
        #return
        use_asyncio_event_loop()
        #asyncio.run(self.my_coroutine())
        loop.run_until_complete(asyncio.gather(
             self.my_coroutine(),
             self.my_draw()))
        # result = loop.run_until_complete(self.my_coroutine())
        # print('result: ', result)
        # x = asyncio.create_task(self.my_coroutine())
