



import datetime


class Logger:


    def __init__(self, level = 'normal'):
        self.thelog = ""
        self.dateline = "date/time: "+ datetime.datetime.now().isoformat() + "\n"
        self.quiet = (level == 'quiet')


    def set_verbosity(self, level):
        """
        (Called by engine. Not called by user.)
        Set the engine to a level of verbosity.
        This only affects messages to stdout.
        Typical use is to make the solver run quietly,
        as it normally generates a lot of stdout messages,
        which are also recorded in the log.

        Arguments:

            level (string):
                Currently level can only be "quiet".

        """
        if level == "quiet":
            self.quiet = True


    def __call__(self, msg, end = None, save = False):
        """
        Arguments:

            msg:
            end:
            save:

        """
        self.log(msg, end, save)


    def log(self, msg, end = None, save = False):
        """
        Arguments:

            msg:
            end:
            save:

        """
        if not save and not self.quiet:
            print(msg, end=end)
        end_ = "\n" if end is None else end
        self.thelog += f"{msg}{end_}"


    def __str__(self):
        return "\n\n\n\n" + self.dateline + "\n" + self.thelog

