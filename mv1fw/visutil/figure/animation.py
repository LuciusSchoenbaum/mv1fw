


import os, glob, re
from PIL import Image




class Animation:
    """
    Helper methods to create animations (.gif, .mp4, ...) without much fuss.
    Wrapped in a class for convenience during standalone processing.

    todo: documentation for duration, frame_count_limit,
    acceptable syntax for frame_glob, ...

    """

    def __init__(self):
        pass


    def from_filename_list(
            self,
            filename_list,
            filename,
            duration=100,
            frame_count_limit=None,
    ):
        """
        Generate an animation using a list of filenames
        pointing to the frames, in the desired order.
        The lowest-level animation-generating routine.

        Arguments:

            filename_list (list of string):
                input frames
            filename (string):
                output filename
            duration (integer):
                duration in ms (todo: check).
                Default: 100
            frame_count_limit (optional integer):
                Optional safety to truncate the list of frames at a
                count-up point.
        """
        frames = [Image.open(image) for image in filename_list]
        frame = frames[0]
        the_rest = frames[1:] if frame_count_limit is None else frames[1:frame_count_limit]
        frame.save(
            filename,
            format="GIF",
            append_images=the_rest,
            save_all=True,
            duration=duration,
            loop=0,
        )


    def from_ordered_image_directory(
            self,
            image_directory,
            filename,
            infiletype=".png",
            duration=100,
            frame_count_limit=None,
    ):
        """
        Make a movie from files in directory
        This method is superseded by the "frame_glob" method below.

        Arguments:

            image_directory:
            filename:
            infiletype:
            duration:
            frame_count_limit:
        """
        # todo sort? review
        filename_list = glob.glob(f"{image_directory}/*{infiletype}")
        self.from_filename_list(
            filename_list = filename_list,
            filename = filename,
            duration = duration,
            frame_count_limit = frame_count_limit,
        )


    def from_frame_glob(
            self,
            frame_glob,
            filename,
            duration=100,
            sort_nicely=False,
            cwd=False,
            frame_count_limit=None,
    ):
        """
        Create an animation file from a glob string indicating the desired frames.


        Arguments:

            frame_glob (string):
                Glob pattern, typ. of the form ...[0-9]+...
                If cwd is False, frame_glob should be a full absolute path.
                If cwd is True, frame_glob can be a filename, for example, foo_[0-9]+.jpg.
            filename:
                Output filename
            duration:
                Duration of each frame, passed to Python PIL Image API.
            sort_nicely:
                Given a list [foo_1.jpg, foo_10.jpg], the default behavior
                (dictionary order, or logical order) will sort as [foo_10.jpg, foo_1.jpg].
                To sort [0-9]+ tagged files the way humans expect, pass sort_nicely=True.
                This behavior is optional. Alternative solutions exist, such as to left-pad
                the frame integers when they are created.
            cwd:
                For kludge-y use case, you do not have to construct an absolute filename.
                Instead, you can direct the current working directory (if necessary)
                and then pass in a path-less filename for the frame_glob.
            frame_count_limit:
                If None: no limit to number of frames.
                If integer N: operation is performed only on the first N frames, at maximum.
                This argument can be used in order to prevent a program from creating
                large, unwieldy, performance-costly files.

        Returns:

            True if an animation was created, otherwise False.

        """
        # > Use Python's glob to grab all the files in some directory,
        # because glob's pattern matching doesn't support [0-9]+.
        if not cwd:
            dirname = os.path.dirname(frame_glob)
            filenames_all = glob.glob(dirname + "/*")
        else:
            # for kludge-y use cases
            filenames_all = glob.glob("*")
        filename_list = [x for x in filenames_all if re.match(frame_glob, x)]
        # print(filenames_all)
        # print(filename_list)
        # exit(0)
        if len(filename_list) == 0:
            return False
        else:
            ok = True
        filename_list = self._sort_nicely(filename_list) if sort_nicely else sorted(filename_list)
        self.from_filename_list(
            filename_list = filename_list,
            filename = filename,
            duration = duration,
            frame_count_limit = frame_count_limit,
        )
        return ok

    def _tryint(self, s):
        try:
            return int(s)
        except:
            return s

    def _sort_nicely(self, l):
        """
        Sort the given list in the way that humans expect.
        """
        alphanum_key = lambda s: [self._tryint(c) for c in re.split('([0-9]+)', s)]
        l.sort(key=alphanum_key)
        return l

