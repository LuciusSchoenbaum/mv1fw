



import numpy as np

from scipy.interpolate import griddata



series_colorcycle = [
    "mediumblue",
    "red",
    "black",
    "seagreen",
    "rebeccapurple",
    "lightsteelblue",
]

# 2: the default, a bold line.
# 1: a slightly thinner line, still visible.
series_linewidth = 1







def interpolate_2D(
        size,
        X,
        in1,
        in2,
        outs,
        method,
        datatype = np.double,
):
    """
    Interpolate values specified on a set of 2D points
    onto a regular mesh whose extent precisely encloses the
    convex hull of the input 2D points.
    Uses the griddata method from scipy.interpolate.

    Note about non-nearest vs. nearest interpolation:
    For methods other than "nearest", the interpolation
    delivered by griddata (scipy's unstructured data interpolation)
    will only find values within the convex hull.
    It will not extend the interpolation outside of that hull in
    the "obvious" way (whatever you think that is) ITCINOOD,
    and the best you can do is modify the
    parameter fill_value which is nan by default.

    Arguments:

        size (integer):
            Create size x size regular mesh to interpolate onto.
        X:
            Input array
        in1 (integer):
        in2 (integer):
            indices into X for x, y axes of the field
        outs (list of integers):
            indices into X for outputs to interpolate
        method (string):
            can be nearest, linear, cubic, ...
        datatype:
            datatype, default numpy double
    """
    # xmin = X[:,in1:in1+1].min()
    # xmax = X[:,in1:in1+1].max()
    # ymin = X[:,in2:in2+1].min()
    # ymax = X[:,in2:in2+1].max()
    xmin = X[:,in1:in1+1].astype(datatype).min()
    xmax = X[:,in1:in1+1].astype(datatype).max()
    ymin = X[:,in2:in2+1].astype(datatype).min()
    ymax = X[:,in2:in2+1].astype(datatype).max()
    extent = [xmin, xmax, ymin, ymax]
    # regular mesh to interpolate onto
    mesh = np.meshgrid(
        np.linspace(extent[0], extent[1], num=size, endpoint=True),
        np.linspace(extent[2], extent[3], num=size, endpoint=True),
        indexing="ij",
    )
    # interpolate outvars onto mesh
    outvars_interp = []
    for out1 in outs:
        outvars_interp.append(griddata(
            points=(X[:,in1], X[:,in2]),
            values=X[:,out1],
            # np.meshgrid produces a list of
            # arrays A, B such that A[i,j],B[i,j] is a point
            # in the set of points "xi". This format is
            # accpetable to scipy griddata, after
            # coercing the list to a tuple datatype.
            # Note: alternatively, griddata will also accept xi's
            # with shape (m,D) where D is the problem dimension.
            xi=tuple(mesh),
            method=method,
        ))
    return extent, outvars_interp











def plot_scatterplot(
        ax,
        title = None,
        X = None,
        x = None,
        xlim = None,
        ylim = None,
        Xs = None,
        xs = None,
        Xslabels=None,
        vlines = None,
        vlineslim = None,
        pointsize = 2 # point^2 (typographic points are 1/72 in.)
):
    """
    Print scatter plot of data in an x-y field.

    Remind: do not recommend calling
    this explicitly - for a special use case,
    implement a "fig" method
    (follow the pattern) and use that instead.

    ideas: (0d)
    - avoid cutoffs at edges of horizontal axis
    - make dots smaller, black
    - remove box frame
    - tick marks at xinit, xfinal
    - boundary given by xinit, xfinal

    Arguments:

        ax (pyplot axis):
            axis to write to
        X (numpy array):
            Plot coordinates ([:,0], [:,i]) for all i > 0 if x is not set,
            otherwise plot (x, [:,i]) for all i ≥ 0.
        x (optional scalar)
            an optional constant x-axis value to pin values to.
        title (string):
            optional title of plot
        xlim (pair of float):
            optional xlimits
        ylim (pair of float):
            optional ylimits
        Xs (list of numpy array):
            Pair (Xs[i], xs[i]) works the same as (X, x) for
            a basic scatterplot. Use if you have more point sets
            you wish to see in the same field.
        xs (list of float):
            See Xs.
        Xslabels (list of string):
            labels to assign to sets in Xs that are plotted
        vlines:
            optional vlines, set ylims to plot vlines.
            Examples: (x1, x2, x3,) or (x1,)
        vlineslim:
            optional limits to vlines, required if vlines != None.
        pointsize:
            size of scatterplot points
    """
    # todo still work in progress.
    sampleset_colors = ["chocolate", "mediumturquoise", "gold", "cornflowerblue"]
    if X is None and x is None:
        ax.yaxis.set_tick_params(labelleft=False)
        ax.yaxis.set_tick_params(left=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    else:
        if x is not None:
            # > the x-axis value is taken to be constant.
            T = np.full((X.shape[0], 1), x)
        else:
            # > the 0th index is interpreted as time (x-axis).
            T = X[:,0:1]
        beg = 1 if x is None else 0
        for i in range(beg, X.shape[1]):
            ax.scatter(T, X[:,i:i+1], s=pointsize, label = "")
    if Xs is not None:
        for i in range(len(Xs)):
            Xi = Xs[i]
            xi = xs[i]
            if Xi.shape[1] == 1 and xi is None:
                setlabel = None if Xslabels is None else Xslabels[i]
                ax.scatter(
                    x=Xi[:,0:1],
                    y=np.zeros_like(Xi[:,0:1]),
                    s=pointsize,
                    label=setlabel,
                )
                ax.yaxis.set_tick_params(labelleft=False)
                if xlim is not None:
                    # an axis that the points can sit on
                    ax.hlines(y=0.0, xmin=xlim[0], xmax=xlim[1], colors='black', linewidths=1.)
            else:
                if xi is not None:
                    T = np.full((Xi.shape[0], 1), xi)
                else:
                    T = Xi[:,0:1]
                beg = 1 if xi is None else 0
                for j in range(beg, Xi.shape[1]):
                    setlabel = None if Xslabels is None else Xslabels[i]
                    ax.scatter(T, Xi[:,j:j+1], s=pointsize, c=sampleset_colors[i%len(sampleset_colors)], label=setlabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if vlines is not None:
        ax.vlines(vlines, vlineslim[0], vlineslim[1], linestyles="dashed", colors="black")
    if title is not None:
        ax.set_title(title)
    if Xslabels is not None:
        # cf. SampleMonitor - this is not having effect. ?
        ax.legend(loc="upper right")






def plot_errorbarseries(
        ax,
        # DATA
        X = None,
        center = None,
        error_idx = None,
        # FORMAT
        title = None,
        inlabel = None,
        outlabel = None,
        inticklabels = None,
        fmt = 'none',
        linewidth = 1,
        capsize = 3,
        xlim = None,
        xticks = None,
        ylim = None,
        yticks = None,
        fmt_999 = False,
):
    """

    Arguments:

        ax:
        title (optional string):
        X (optional numpy array):
            A numpy array of the form [x, y, y-, y+],
            listing the x coordinates, y coordinates,
            negative error in y (in absolute value), and positive error in y,
            as columns, thus an array of shape (N, 4).
            If ``X`` is not None, this input overrides inputs x, y, errminus, errplus.
        center (optional scalar):
            Pass the average over all x of the responses y, if
            it is desired that this value is emphasized in the generated plot.
        error_idx (optional integer):
        inlabel (optional string):
        outlabel (optional string):
        inticklabels (optional list of string):
        fmt (string):
            Pyplot format string. Examples: '-', 'o', 'o-'.
        linewidth (scalar):
        capsize (optional scalar):
        xlim: (optional pair of scalar)
        xticks (:
        ylim:
        yticks:
        fmt_999 (boolean):
            Format in a standard way for 3-confidence level plot,
            overrides other formatting.

    """
    # the control via arguments is a little out of order (apologies),
    # but we're proceeding on an as-needed basis and we are
    # a little time-constrained. Tweak if necessary.

    xmin = 0
    xmax = X.shape[0]+1
    x = np.arange(1, xmax)
    y = X[:,1]
    if X.shape[1] > 2:
        # todo awk condition
        if error_idx is None:
            yinterval = X[:,2:].max()
        else:
            yinterval = X[:,error_idx:error_idx+2].max()
    else:
        # > no error bars
        yinterval = 0.0
    ymin = (y.min()-yinterval)
    ymax = (y.max()+yinterval)
    # 20%
    ypad1 = abs(ymax-ymin)*0.2
    ymin -= ypad1
    ymax += ypad1
    xlim = (xmin, xmax)
    ylim = (ymin, ymax)
    # don't tick the endpoints of the x range
    xticks = np.arange(xmin+1,xmax)
    ax.set(
        xlim=xlim,
        xticks=xticks,
        ylim=ylim,
        # todo something wrong?
        # yticks=yticks,
        title=title,
        xlabel=inlabel,
        ylabel=outlabel,
    )
    if inticklabels is not None:
        maxticklength = 0
        for tick in inticklabels:
            lentick = len(tick)
            if lentick > maxticklength:
                maxticklength = lentick
        if x.shape[0] <= 6 and maxticklength <= 4:
            # > no rotation
            ax.set_xticks(ticks=x, labels=inticklabels)
        else:
            ax.set_xticks(ticks=x, labels=inticklabels, rotation=45)
    # > hide top and right borders
    for direction in ["right", "top"]:
        ax.axis[direction].set_visible(False)
    bar_width = 0.5
    if center is not None:
        ax.hlines(
            y=center,
            xmin=xmin,
            xmax=xmax,
            color='red',
            linewidth=1,
        )
        ax.bar(
            x=x,
            height=y - np.full_like(y, center),
            width=bar_width,
            bottom=center,
            color='cornflowerblue',
        )
    else:
        ax.bar(
            x=x,
            height=y-np.full_like(y, ymin),
            width=bar_width,
            bottom=ymin,
            color='cornflowerblue',
        )
    if fmt_999:
        capsize999 = 6
        linewidth999 = 4
        ax.errorbar(
            # DATA:
            x=x,
            y=y,
            yerr=X[:,2:4].transpose(),
            # FORMAT:
            fmt='none',
            linewidth=linewidth999,
            capsize=None,
            elinewidth=linewidth999,
            color='black',
            # fillstyle='left',
        )
        ax.errorbar(
            x=x,
            y=y,
            yerr=X[:,4:6].transpose(),
            fmt='none',
            linewidth=0.5*linewidth999,
            # elinewidth=linewidth,
            capsize=0.5*capsize999,
            color='black',
        )
        ax.errorbar(
            x=x,
            y=y,
            yerr=X[:,6:8].transpose(),
            fmt='none',
            linewidth=0.2*linewidth999,
            capsize=5.0,
            color='black',
        )
    else:
        # > only add error bars if there is data for them
        if X.shape[1] > 2:
            eidx = error_idx if error_idx is not None else 2
            ax.errorbar(
                # DATA:
                x=x,
                y=y,
                yerr=X[:,eidx:eidx+2].transpose(),
                # FORMAT:
                fmt=fmt,
                linewidth=1,
                capsize=5.0,
                color='black',
                # fillstyle='left',
            )




def plot_multiseries(
        ax,
        title,
        XYs,
        inlabel,
        outlabels,
        xlim,
        ylim,
        marker = "x",
):
    """
        Plot varying input sequences X (in a common dimension),
        versus varying output sequences Y (in a common dimension),
        or in other words, a list (XY1, XY2, ...) of sets XY of (x,y) pairs.
        This is referred to as a "multiseries", whereas a "series" (XY1Y2Y3...)
        has a common set of inputs X common to all output sequences Y.

        Arguments:

            ax:
            title:
            XYs:
                List of arrays XY = [x, y]. Each XY of size Nx2
                associates input value x to value/output/response y,
                N may vary between arrays, and
                x sequence may vary between arrays.
                It is assumed that the x sequence increases down the array.
            inlabel:
            outlabels:
            xlim:
            ylim:
            marker:
                todo set it up optional/etc.
    """
    for i, XY in enumerate(XYs):
        ax.plot(
            XY[:,0],
            XY[:,1],
            label=outlabels[i],
            color=series_colorcycle[i%len(XYs)],
            marker=marker,
            linewidth=series_linewidth,
        )
    ax.set_xlabel(inlabel)
    if len(outlabels) == 1:
        ax.set_ylabel(outlabels[0])
    else:
        # show the legend
        ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title)









def plot_series(
        ax,
        title,
        X,
        inlabel,
        inidx,
        outlabels,
        outidxs,
        t=None,
        xlim=None,
        ylim=None,
        vlines=None,
        vlineslim=None,
        X_black=None,
        X_ref=None,
        half_linestyle=None,
        half_color=None,
        marker=None,
        half_marker=None,
):
    """
    Plot a time series-style plot, x-y == inlabel-outlabel.
    Optional t only modifies title.

    All references are colored red, which may
    not be desired behavior for a high number of outputs.
    However, we don't see a pressing need to modify at this time.

    Arguments:

        ax:
            requested Matplotlib axis for plotting
        X:
            Array of shape [n, m] of points in the plane,
            where n is the number of points, and m ≥ 2.
            It may be that m = 2 and the array represents n points in the plane.
            Otherwise it represents m - 1 sets of n points in the plane.
            NOTE: ndarray must be sorted along the "inidx" index (typ. 0).
        inlabel (string):
            label for input (x axis)
        inidx (integer):
            index for input, almost always 0.
        outlabels (list of string):
            list of labels for output.
        outidxs (list of integer):
            list of indices in ndarray to plot as outputs.
        title (string):
            A title string you may construct
        t:
            optional time for a plot of a timeslice, it will modify the title.
        xlim:
            optional xlimits list: (bottom, top).
        ylim:
            optional ylimits list: (bottom, top).
        vlines:
            optional vlines, set ylims to plot vlines.
            Examples: (x1, x2, x3,) or (x1,)
        vlineslim:
            optional limits to vlines, required if vlines != None.
        X_black:
            Additional plot series can be passed in here, they will be printed in black.
        X_ref:
            Optional ndarray, contains a reference curve to the
            main series ndarray.
            Ensure it is the same shape as X.
        half_linestyle:
            Optional string or parametrized linestyle,
            whether to repeat the colors on the N = 2n inputs
            (if N is odd, the last input will dangle) and instead distinguish via
            the indicated line style.
            Can be used for reference data plots. String can be:
            {"solid", "dotted", "dashed", "dashdot"}.
            See pyplot documentation for parametrized linestyles or official docs.
            Default: None
        half_color:
            Optional string, color of latter half of plotted curves.
            Default: None
        marker:
            Optional character or integer indicating a marker for individual data points.
            Character an be:
            {'x', '+', '1', '2', '3', '4', '_', '|'}
            See pyplot documentation for integer markers or official docs.
            Default: None
        half_marker:
            Optional character or integer indicating a marker for individual data points,
            to be used on the back half. (See marker argument.)
            In order to be applied, half_linestyle must be set, otherwise
            the series are not halved.
    """
    if half_linestyle is not None or half_color is not None:
        len1 = int(len(outidxs)/2.0)
        len2 = len(outidxs) - len1
    else:
        len1 = len(outidxs)
        len2 = None
    for i in range(len1):
        ax.plot(
            X[:,inidx],
            X[:,outidxs[i]],
            label=outlabels[i],
            color=series_colorcycle[i%len(series_colorcycle)],
            marker=marker,
            linewidth=series_linewidth,
        )
    if half_linestyle is not None:
        for i in range(len2):
            ax.plot(
                X[:,inidx],
                X[:,outidxs[len1+i]],
                label=outlabels[len1+i],
                color=half_color if half_color is not None else series_colorcycle[(i+len1)%len(series_colorcycle)],
                linestyle=half_linestyle,
                marker=half_marker if half_marker is not None else marker,
                linewidth=series_linewidth,
            )
    if X_ref is not None:
        for i in range(len(outidxs)):
            ax.plot(
                X_ref[:,inidx],
                X_ref[:,outidxs[i]],
                label=outlabels[i]+"_ref",
                color="red",
                linewidth=series_linewidth,
            )
    if X_black is not None:
        for i in range(1,X_black.shape[1]):
            ax.plot(X_black[:,0], X_black[:,i], color="black")
    ax.set_xlabel(inlabel)
    if len(outlabels) == 1:
        ax.set_ylabel(outlabels[0])
    else:
        # show the legend
        ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if vlines is not None:
        ax.vlines(vlines, vlineslim[0], vlineslim[1], linestyles="dashed", colors="black")
    if title is not None or t is not None:
        title_ = f"{title}, t = {t:.4f}" if title is not None and t is not None else f"t = {t:.4f}" if title is None and t is not None else title
        ax.set_title(title_)






def plot_heatmap(
        ax,
        title,
        X,
        in1,
        in2,
        out1,
        lbl,
        value_range,
        method,
        size_interpolate_2D,
        t=None,
        xlim=None,
        ylim=None,
        vlines = None,
        vlineslim = None,
        X_black = None,
        X_black_linewidth = None,
        # X_ref = None,
        cmap = "plasma",
        datatype = np.double,
):
    """
    Plot as heatmap.
    Create xy field using index x=in1, y=in2,
    reading into X.
    Use output variable u=out1 for color/heat.
    Pass in a consistent value_range if desired.

    Arguments:

        ax (Matplotlib axis):
        X (array):
        in1 (integer):
            index of X
        in2 (integer):
            index of X
        out1 (integer):
            index of X
        lbl (list of string):
        title (string):
        value_range (optional [lower,upper]):
            Pass in value_range if a consistent colorbar
            (assignment of color to value) is needed.
        method (string):
            method of interpolation. Can be:
            antialiased, nearest, linear, bicubic, ...
        t:
            optional time for a plot of a timeslice, it will modify the title.
        xlim:
            optional xlimits list: (bottom, top).
        ylim:
            optional ylimits list: (bottom, top).
        vlines:
            optional vlines, set ylims to plot vlines.
            Examples: (x1, x2, x3,) or (x1,)
        vlineslim:
            optional limits to vlines, required if vlines != None.
        X_black:
            values to plot on the heatmap as black lines. The 0th index
            is used for x values and other indices are used for y values.
        X_black_linewidth (float):
            width of X_black lines in points.
        cmap:
            colormap, e.g. "viridis", "plasma", ...options are numerous, see
            matplotlib documentation.
        size_interpolate_2D:
            integer giving the size of the regular mesh (roughly, "pixel values")
            that is used to create the figure.
            100 is usually ok, but be careful not to pixellate unnecessarily
            if you have a lot of data to work with. Default: 400 ITCINOOD.
        datatype:
            Numpy datatype, default numpy.double
    """
    size_interpolate_2D_default = 400
    extent, interps = interpolate_2D(
        size=size_interpolate_2D if size_interpolate_2D is not None else size_interpolate_2D_default,
        X=X,
        in1=in1,
        in2=in2,
        outs=[out1],
        method=method,
    )
    # set vmin, vmax explicitly for consistent color assignments
    vmin_ = value_range[0] if value_range is not None else None
    vmax_ = value_range[1] if value_range is not None else None
    im = ax.imshow(
        interps[0].T.astype(datatype),
        origin="lower",
        extent=tuple(extent),
        vmin=vmin_,
        vmax=vmax_,
        # default aspect is 'equal', this means aspect ratio = 1.
        # This can give an oblong rectangle plot.
        # Using 'auto' (or a custom float...) will fill out the
        # space (the Axes in pyplot lingo),
        # but be advised, pixels will not be square.
        aspect='auto',
        cmap=cmap,
    )
    if lbl is not None:
        ax.set_xlabel(lbl[in1])
        ax.set_ylabel(lbl[in2])
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if vlines is not None:
        ax.vlines(vlines, vlineslim[0], vlineslim[1], linestyles="dashed", colors="black")
    if X_black is not None:
        for i in range(1,X_black.shape[1]):
            ax.plot(X_black[:,0], X_black[:,i], color="black", linewidth=X_black_linewidth)
    if title is not None or t is not None:
        title_ = f"{title}, t = {t:.4f}" if title is not None and t is not None else f"t = {t:.4f}" if title is None and t is not None else title
        ax.set_title(title_)
    return im






def plot_scatterplot3d(
        ax,
        Xs,
        in1,
        in2,
        in3,
        lbl,
        title = None,
        pointsize = 2, # point^2 (typographic points are 1/72 in.)
        sampleset_colors = None,
):
    length = len(Xs)
    for i in range(length):
        X = Xs[i]
        ax.scatter(
            xs=X[:,in1],
            ys=X[:,in2],
            zs=X[:,in3],
            s=pointsize,
            c=None if sampleset_colors is None else sampleset_colors[i%length],
        )
    ax.set_xlabel(lbl[in1])
    ax.set_ylabel(lbl[in2])
    ax.set_zlabel(lbl[in3])
    if title is not None:
        ax.set_title(title)









