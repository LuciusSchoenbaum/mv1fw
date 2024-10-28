




from os.path import \
    join as os_path_join
from os import \
    getcwd as os_getcwd
from shutil import \
    rmtree as shutil_rmtree
from math import log2

from .types import (
    tag_filename,
    width10,
)
from .ossys import create_dir




class CogManager:
    """
    Class to perfom system calls
    that modify the file system,
    and to generate adequately descriptive filenames
    for results, data, plots, etc.
    It is called "CogManager", perhaps short for "CognomenManager"
    which implies both a name and a family (directory tree) context,
    or implying that it produces "little cogs in the machine".
    It is not a very descriptive name, but it is a more convenient name
    than, say, "DirectoryTreeAndFilenameForArtifactManager".

    The caller/invoker is responsible for establishing
    the root directory where files and directories will be placed,
    and in particular, whether that directory is clean or not.
    There is the option to pass None, and in that case,
    a directory 'output' in the current working directory is used.

    Arguments:

        output_abs_path_root (optional string):
            When this is passed the output_stem variable
            is joined to this (absolute) path to create the
            root path for storing all artifacts.

    """

    # todo note: the class is *somewhat* general.
    #  I do think some things about it are correct.
    #  It grew out of a utility in PyPinnch and carries some
    #  ideas vestigially.
    #  and it is used by QueueG to do general things.
    #  It can be improved.


    def __init__(
            self,
            output_abs_path_root = None,
    ):
        # these three directories are all different, in general,
        # but all the same if there is one driver, and one phase.
        # this allows binning of results from different drivers and different phases.
        self.engine_dir = output_abs_path_root
        self.driver_dir = None
        self.phase_dir = None
        # the number of strides in the entire run, same as engine.topline.stride
        self.stride = None

        # widths

        # step counter width
        self.tiwidth = None
        # model counter width
        self.nniwidth = None
        # driver counter width
        self.driwidth = None
        # phase counter width
        self.phiwidth = None
        # level counter width
        self.Lwidth = None
        # timestep counter width
        self.tjwidth = None
        # timeslice counter width
        self.sjwidth = None
        # substep counter width
        self.skwidth = None
        # iteration counter width
        self.itwidth = None
        # train counter width
        self.trwidth = None
        # sl counter width
        self.slwidth = None

        # endings

        self.plot = ".png"
        self.dat = ".dat"
        self.vtk = ".vtp"
        self.info = ".txt"
        self.chk = ".pt"

        # stems

        self.data_stem = "dat"
        self.vtk_stem = "vtk"
        self.fig_stem = "fig"
        self.anim_stem = "anim"
        self.checkpoint_stem = "checkpoint"
        self.probe_stem = "probe"
        self.post_stem = "post"



    def init(
            self,
            engine,
    ):
        """
        Called by engine at start of routine, "on start".
        This sets widths and creates directories.

        Arguments:

            engine (:any:`Engine`):

        """
        # > check for an underspecified run
        if self.engine_dir is None:
            # > dump into the local directory
            # > use "output" subdir as an extra precaution
            # This is not the recommended way,
            # but a possible way to proceed, mainly
            # for cases where the goal is to run using
            # the most minimal amount of code possible.
            self.engine_dir = create_dir(os_getcwd(), ["output"])
        self.tiwidth = width10(engine.problem.th.Nstep())
        self.stride = engine.topline.stride
        nmodels = len(engine.models)
        if nmodels <= 1:
            self.nniwidth = None
        else:
            self.nniwidth = width10(nmodels)
        ndriver = len(engine.drivers)
        if ndriver <= 1:
            self.driwidth = None
        else:
            self.driwidth = width10(ndriver)
        substepmax = 1
        for labels in engine.problem.solutions:
            sol = engine.problem.solutions[labels]
            substep_ = sol.substep
            if substep_ > substepmax:
                substepmax = substep_
        self.skwidth = width10(substepmax)
        # > point of no return - clean the output tree and prepare it for new output
        self.clean_tree()
        for action in engine.actions:
            create_dir(
                base_dir=self.engine_dir,
                stem=str(action),
            )
        for probe in engine.probes:
            create_dir(
                base_dir=self.engine_dir,
                stem=str(probe),
            )



    def init_stride(
            self,
            driver,
            ti,
    ):
        """
        Set up output at beginning a stride

        NOTE: prior to parallel driver update,
        it can be assumed that the driver is unique.

        Arguments:

            driver:
            ti:

        """
        nphase = len(driver.phases)
        if nphase <= 1:
            self.phiwidth = None
        else:
            self.phiwidth = width10(nphase)
        # stride directory - each stride has several driver's training/result/etc output
        # ti_marker: _ti0, remove the underscore
        if self.stride > 1:
            # > distinguish driver's output by indicating
            # the beginning of the stride, using the ti the counter.
            stem = self.ti_marker(ti)[1:]
            self.driver_dir = create_dir(self.engine_dir, stem)
        else:
            # > one stride - simplify the tree structure
            self.driver_dir = self.engine_dir
        # > create locations for driver actions and probes
        for action in driver.actions:
            create_dir(
                base_dir=self.driver_dir,
                stem=str(action),
            )
        for probe in driver.probes:
            create_dir(
                base_dir=self.driver_dir,
                stem=str(probe),
            )


    def init_phase(
            self,
            phase,
            phi,
    ):
        """
        Called at beginning of each phase
        to config pasta for generating artifacts for the phase.

        Arguments:

            phase:
            phi:

        """
        nstep = phase.th.Nstep()
        grading_using = phase.strategies.using('grading')
        if phase.strategies.using('optimizer'):
            max_iterations = phase.strategies.optimizer.kit.max_iterations
        else:
            max_iterations = 1
        self.tjwidth = width10(nstep)
        self.sjwidth = width10(nstep+1)
        self.trwidth = width10(2*nstep)
        if grading_using:
            depth = int(log2(nstep))
            self.Lwidth = width10(depth)
        else:
            self.Lwidth = None
        self.itwidth = width10(max_iterations)
        stem = self.ph_marker(phi)
        stem = None if stem == "" else stem # awk
        self.phase_dir = create_dir(self.driver_dir, stem)
        for action in phase.actions:
            create_dir(
                base_dir=self.phase_dir,
                stem=str(action),
            )
        for probe in phase.probes:
            create_dir(
                base_dir=self.phase_dir,
                stem=str(probe),
            )


    def ti_marker(self, ti):
        if ti is None:
            return ""
        if ti == "*":
            return ".ti[0-9]+"
        return ".ti" + str(ti).zfill(self.tiwidth)

    def tj_marker(self, tj):
        if tj is None:
            return ""
        if tj == "*":
            return ".tj[0-9]+"
        return ".tj" + str(tj).zfill(self.tjwidth)

    # todo review sj marker
    def sj_marker(self, sj, sk):
        # sj marker is exceptional after the .mv1 format
        # was added to the project. This is irregular
        # and it may be smoothed over at some point,
        # but seems to be doing no harm.
        #
        # sk marker only used with sj marker.
        # cf. Action::Result
        #
        if sj is None:
            out = ""
        elif sj == "*":
            out = ".t[0-9]+"
        else:
            out = ".t" + str(sj).zfill(self.sjwidth)
        if sk is not None:
            if sk == "*":
                out += ".[0-9]+"
            elif sk is not None:
                out += "." + str(sk).zfill(self.skwidth)
        return out

    # def sk_marker(self, sk):
    #     if sk is None:
    #         return ""
    #     if sk == "*":
    #         return ".[0-9]+"
    #     return "." + str(sk).zfill(self.skwidth)

    def it_marker(self, it):
        if it is None:
            return ""
        if it == "*":
            return ".it[0-9]+"
        return ".it" + str(it).zfill(self.itwidth)

    def tr_marker(self, tr):
        if tr is None:
            return ""
        if tr == "*":
            return ".tr[0-9]+"
        return ".tr" + str(tr).zfill(self.trwidth)

    def L_marker(self, L):
        if L is None or self.Lwidth is None:
            return ""
        if L == "*":
            return ".L[0-9]+"
        return ".L" + str(L).zfill(self.Lwidth)

    def sl_marker(self, sl):
        if sl is None:
            return ""
        return ".sl" + str(sl).zfill(self.slwidth)

    # NOTE: driver, phase, model, level:
    # Cog will suppress a trivial tab -
    # you don't have to worry about it, just loop over things::
    #       width == None
    # is a signal that there's only one of that thing.

    def dri_marker(self, dri):
        if dri is None or self.driwidth is None:
            return ""
        return ".dri" + str(dri).zfill(self.driwidth)

    def ph_marker(self, phi):
        if phi is None or self.phiwidth is None:
            return ""
        return ".ph" + str(phi).zfill(self.phiwidth)

    def ni_marker(self, modeli):
        if modeli is None or self.nniwidth is None:
            return ""
        return ".ni" + str(modeli).zfill(self.nniwidth)


    #<><><><><><><><><><><><><><>


    def ti_title(self, ti, width = True):
        if ti is None:
            return ""
        if width:
            return "ti " + str(ti).zfill(self.tiwidth) + " "
        else:
            return f"ti {ti} "

    def tj_title(self, tj, width = True):
        if tj is None:
            return ""
        if width:
            return "tj " + str(tj).zfill(self.tjwidth) + " "
        else:
            return f"tj {tj} "

    # time slice counter
    def sj_title(self, sj, sk = None, width = True):
        if sj is None:
            return ""
        if width:
            out = "sj " + str(sj).zfill(self.sjwidth)
        else:
            out = f"sj {sj} "
        if sk is None:
            out += " "
        else:
            out += "." + str(sk).zfill(self.skwidth) + " " if width else f".{sk} "
        return out

    # training step counter
    def it_title(self, it, width = True):
        if it is None:
            return ""
        if width:
            return "it " + str(it).zfill(self.itwidth) + " "
        else:
            return f"it {it} "

    # training loop counter
    def tr_title(self, tr, width = True):
        if tr is None:
            return ""
        if width:
            return "tr " + str(tr).zfill(self.trwidth) + " "
        else:
            return f"tr {tr} "

    # level counter
    def L_title(self, L, width = True):
        if L is None or self.Lwidth is None:
            return ""
        if width:
            return "L " + str(L).zfill(self.Lwidth) + " "
        else:
            return f"L {L} "

    # driver counter
    def dri_title(self, dri, width = True):
        if dri is None or self.driwidth is None:
            return ""
        if width:
            return "dri " + str(dri).zfill(self.driwidth) + " "
        else:
            return f"dri {dri} "

    # phase counter
    def ph_title(self, ph, width = True):
        if ph is None or self.phiwidth is None:
            return ""
        if width:
            return "ph " + str(ph).zfill(self.phiwidth) + " "
        else:
            return f"ph {ph} "

    # model counter
    def ni_title(self, modeli, width = True):
        if modeli is None or self.nniwidth is None:
            return ""
        if width:
            return "ni " + str(modeli).zfill(self.nniwidth) + " "
        else:
            return f"ni {modeli} "


    #<><><><><><><><><><><><><><><>


    def _filename_noending(
            self,
            handle,
            driveri,
            phasei,
            modeli,
            ti,
            tj,
            sj,
            sk,
            tr,
            it,
            level,
            sl,
            tag,
    ):
        """
        Filename builder
        """
        out = handle \
              + self.dri_marker(driveri) \
              + self.ph_marker(phasei) \
              + self.ni_marker(modeli) \
              + self.ti_marker(ti) \
              + self.tj_marker(tj) \
              + self.sj_marker(sj, sk) \
              + self.tr_marker(tr) \
              + self.it_marker(it) \
              + self.L_marker(level) \
              + self.sl_marker(sl)
        if tag is not None:
            out += "." + tag
        return out


    def filename(
            self,
            handle,
            action=None,
            stem=None,
            ending=None,
            driveri=None,
            phasei=None,
            modeli=None,
            ti=None,
            tj=None,
            sj=None,
            sk=None,
            tr=None,
            it=None,
            level=None,
            sl=None,
            tag=None,
    ):
        """
        Create a filename of the
        of the described type, marked by
        stride, step, instep, level, and tag (if any).
        You can use any argument like a keyword argument
        and the associated value will appear in the filename,
        and arguments you do not name explicitly
        will not appear in the filename.
        Anything that doesn't appear explicitly can be set manually
        as a "tag".

        Arguments:

            handle (string):
                Base of filename that will be built.
            action (optional :any:`Action`):
                Always pass self when working inside of an action,
                whereas if pasta is used standalone, this argument can be ignored.
            modeli (string):
                todo i think optional
            driveri (string):
                todo i think optional
            phasei (string):
                todo i think optional
            stem (string):
                todo i think optional
            ending (string):
                todo optional?
            ti (integer or "*"):
                the timestep progress up to the current stride
                (a way of marking the stride)
            tj (integer or "*"):
                the timestep progress within the current stride,
                measured in the stepsize of the current phase.
            sj (integer or "*"):
                the timeslice counter, an alternative to tj
                if you need to produce n+1 artifacts (one for the initial state)
            sk (integer):
                the substep (substep-of-a-timeslice) counter
            tr (integer or "*"):
                training counter (train() calls during phase)
            it (integer or "*"):
            level (integer):
                Any other information used to distinguish the filename.
                The tag is printed verbatim in the filename.
            sl (integer):
            tag (string):

        Returns:

            An absolute path to a file in the run's output directory tree

        """
        # todo it is more stable now - add more documentation
        if action is None:
            if stem is not None:
                # if not self.post, you may use the stem.
                # todo change later?
                tgt_dir = os_path_join(self.engine_dir, stem)
                create_dir(tgt_dir)
            else:
                tgt_dir = self.engine_dir
            cog_id = 0
        else:
            stems = [action.__class__.__name__]
            cog_id = action.cog_id
            tgt_dir = self.engine_dir if cog_id == 0 else self.driver_dir if cog_id == 1 else self.phase_dir
            stems.append(stem)
            tgt_dir = create_dir(tgt_dir, stems)
        filename = self._filename_noending(
            handle=handle,
            driveri=None if cog_id > 0 else driveri,
            phasei=None if cog_id > 1 else phasei,
            modeli=modeli,
            # In a driver dir (pid 1) or phase dir (pid 2),
            # ti is indicated by the directory.
            ti=None if cog_id > 0 else ti,
            tj=tj,
            sj=sj,
            sk=sk,
            tr=tr,
            it=it,
            level=level,
            sl=sl,
            tag=tag,
        )
        # perhaps this can be improved.
        if ending is not None:
            filename += "." + ending
        else:
            if stem is not None:
                filename += "." + stem
        pathname = os_path_join(tgt_dir, filename)
        return pathname


    def title(
        self,
        width=True,
        driveri=None,
        phasei=None,
        modeli=None,
        ti=None,
        tj=None,
        sj=None,
        sk=None,
        tr=None,
        it=None,
        level=None,
        tag=None,
    ):
        """
        Frame title builder
        Use widths to keep animation frames consistent.
        """
        out = self.dri_title(driveri, width) \
              + self.ph_title(phasei, width) \
              + self.ni_title(modeli, width) \
              + self.ti_title(ti, width) \
              + self.tj_title(tj, width) \
              + self.sj_title(sj, sk, width) \
              + self.tr_title(tr, width) \
              + self.it_title(it, width) \
              + self.L_title(level, width)
        if tag is not None:
            out += tag
        return out


    def tag_filename(self, filename, insert):
        """
        Tag a filename after it has already been constructed.
        Useful when multiple filenames are needed that
        relate to one another.

        Arguments:
            filename (string):
                a filename, not modified in place.
            insert (string):
                a tag to insert, before the ending.
        Returns:
            string
        """
        return tag_filename(filename, insert)


    def tag_title(self, title, insert):
        """
        Helper for tagging a title.

        Arguments:

            title: (string)
            insert: (string)
        """
        return title + " " + insert


    # todo deprecated. In light of experience, a simpler system
    #  of using runtags is far better for everyone involved. If the user
    #  wishes to do XYZ (make repeated runs, make sweeps, experiments, ...)
    #  external utilities are responsible.
    # def get_run_stem(self, handle, runtag):
    #     """
    #     Create the stem directory for output.
    #     Typ. a directory /output is built, a directory
    #     like xyz1234 is added in that directory,
    #     and engine_dir is then output/xyz1234.
    #
    #     Arguments:
    #
    #         handle: handle
    #         runtag: tag for multiple runs
    #     """
    #     if handle is not None:
    #         stem = handle
    #         if runtag is not None:
    #             stem += "." + runtag if runtag != "" else ""
    #         tmp = os.path.join(self.engine_dir, stem)
    #         if os.path.exists(tmp):
    #             if self.clean:
    #                 shutil.rmtree(tmp)
    #             else:
    #                 ri = 1
    #                 safety = 50
    #                 while True:
    #                     stem += "_run" + str(ri).zfill(8)
    #                     tmp = os.path.join(self.engine_dir, stem)
    #                     if os.path.exists(tmp):
    #                         ri += 1
    #                         if ri == safety:
    #                             raise ValueError(f"Detected {safety} previous runs! Clean runs before proceeding?")
    #                     else:
    #                         break
    #     else:
    #         stem = None
    #     return stem


    def create_subdir(self, stem):
        """
        Standalone instance helper.
        Unceremoniously create subdirectories.

        Arguments:

            stem (string): subdirectory of engine/main directory,
                or a list of stems, each of which will be created
                on the fly if necessary.

        """
        create_dir(self.engine_dir, stem)


    def clean_tree(self):
        """
        Use a shell utility to clean the output directory.

        """
        # > the point of no return - clean the output directory
        shutil_rmtree(self.engine_dir)


    def path(
            self,
            action = None,
    ):
        """
        Return the output path,
        optionally extended into an
        action's subfolder.

        Arguments:

            action (optional :any:`Action`):
                an optional action, in case the root path of the
                action (the caller) is sought.

        Returns:

            string: path name

        """
        if action is not None:
            out = os_path_join(self.engine_dir, action.__class__.__name__)
        else:
            out = self.engine_dir
        return out

    def root_path(self):
        """
        Return the path without including the `Location` runtag.

        Returns:

            string: path name

        """
        out = "123"
        return out

    # todo this is certainly incorrectly placed, cf StrideMonitor
    def init_slcounter(
            self,
            nslice = None,
    ):
        """
        Deprecated.
        Called by :any:`StrideMonitor` if using slices.

        Arguments:

            nslice (integer): Number of slices

        """
        self.slwidth = 1 if nslice is None else width10(nslice)

