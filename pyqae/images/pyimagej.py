import subprocess
import threading


class RunCmd(threading.Thread):
    '''RunCmd Class
        This class allow for running a command line program using threads
        Also implement a timeout option that allow for killing the process if
        it takes longer than timeout
    '''

    def __init__(self, cmd, timeout):
        threading.Thread.__init__(self)
        self.cmd = cmd
        self.timeout = timeout
        self.p = None

    def run(self):
        self.p = subprocess.Popen(self.cmd)
        self.p.wait()

    def Run(self):
        self.start()
        self.join(self.timeout)

        if self.is_alive():
            self.p.kill()  # use self.p.terminate() if process dont needs a kill -9
            self.join()


class ImageJ(object):
    '''imagej Class
        This class is an abstraction for an instance of ImageJ
        It allows for calling macros from a python script
    '''

    def __init__(self, imagej_exe):
        assert imagej_exe != '', "[ERROR]: ImageJ path is empty"
        self._executable = imagej_exe

    def runImageJMacro(self, macroPath, macroParams, timeout=60, macroName='Macro'):
        '''Run a ImageJ Macro using command line

        Keyword arguments:
        macroPath                      -- Path to macro file *.ijm (mandatory)
        macroParams                    -- Parameters that the macro uses (mandatory)
        msgOpt [\'Running Macro ...\'] -- Message for debugging the will be shown when macro is running

        '''

        cmdline = self._executable + ' --no-splash -macro  \"' + macroPath + '\" \"' + macroParams + '\"'

        RunCmd(cmdline, timeout).Run()
