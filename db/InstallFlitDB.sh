#!/bin/bash
TAGS=(setup_db_host.sh  matplotlibrc  tables.sql  )
PREFIX=18370
#This file was auto-generated

#This is the FLiT DB installer

for x in $(seq 0 $((${#TAGS[@]}-2)));
do
    begin=$(grep --line-number ^${PREFIX}${TAGS[$x]}$ $0 | cut -d ':' -f 1)
    end=$(grep --line-number ^${PREFIX}${TAGS[$(($x + 1))]}$ $0 | cut -d ':' -f 1)
    sed -n $((begin + 1)),$((end - 1))p $0 > ${TAGS[$x]#${PREFIX}}
done

tail -n +$((end + 1)) $0 > ${TAGS[$((${#TAGS[@]}-1))]#${PREFIX}}

EXE=${TAGS[0]#${PREFIX}}

chmod 775 $EXE

./$EXE

exit
18370setup_db_host.sh
#!/bin/bash

set -x

exists ()
{
    command -v "$1" >/dev/null 2>&1
}

python3_has ()
{
    python3 -c "import $1" >/dev/null 2>&1
}

SCRIPT_DIR="$(pwd)/$(dirname $0)"

# Check for psql install
if  ! exists createdb || ! exists psql; then
    # Install if not present
    echo "Postgres does not seem to be installed."
    echo "Attempting install now."

    # Try different package managers
    if exists apt; then
	sudo apt install postgresql postgresql-plpython3
    elif exists apt-get; then
	sudo apt install postgresql postgresql-plpython3
    elif exists pacman; then
	sudo pacman -S postgresql postgresql-lib-python3
    elif exists yum; then
	sudo yum install postgresql-server #name-for-plpython3
    elif exists brew; then
	brew install postgresql --with-python3
	brew services start postgresql
    else
	echo "Unable to find a suitable package manager."
	echo "Please install Postgres and plpython3"
	exit -1
    fi
fi


# Check for numpy install
if  ! python3_has numpy; then
    # Install if not present
    echo "Numpy does not seem to be installed for python 3."
    echo "Attempting install now."

    Try different package managers
    if exists apt; then
    	sudo apt install python3-numpy
    elif exists apt-get; then
    	sudo apt install python3-numpy
    elif exists pacman; then
    	sudo pacman -S python3-numpy
    elif exists yum; then
    	sudo yum install python3-numpy
    elif exists brew; then
    	brew install numpy -with-ptyhon3
    else
    	echo "Unable to find a suitable package manager."
    	echo "Please install numpy for python3"
    	exit -1
    fi
fi


# Check for matplotlib install
if  ! python3_has matplotlib; then
    # Install if not present
    echo "Matplotlib does not seem to be installed for python 3."
    echo "Attempting install now."

    Try different package managers
    if exists apt; then
    	sudo apt install python3-matplotlib
    elif exists apt-get; then
    	sudo apt install python3-matplotlib
    elif exists pacman; then
    	sudo pacman -S python3-matplotlib
    elif exists yum; then
    	sudo yum install python3-matplotlib
    elif exists brew; then
	brew tap homebrew/science
    	brew install homebrew/science/matplotlib -with-ptyhon3
    else
    	echo "Unable to find a suitable package manager."
    	echo "Please install Postgres and plpython3"
    	exit -1
    fi
fi

# Check if user exists
# from http://stackoverflow.com/questions/8546759/how-to-check-if-a-postgres-user-exists
if psql -t -c '\du' | cut -d \| -f 1 | grep -qw `whoami`; then
    echo "User `whoami` already exists"
else
    echo "Creating user `whoami`"
    sudo -u postgres createuser --superuser `whoami`
fi


createdb flit "The database for collecting all FLiT results"
psql flit < "$SCRIPT_DIR/tables.sql"

wait

#add our config to postgres for matplotlib
PGDIR=$(psql flit -t -c 'select getpwd()')
if [ ! -e ${PGDIR}/matplotlibrc ]; then
    sudo -u postgres cp ${SCRIPT_DIR}/matplotlibrc ${PGDIR}/matplotlibrc
else
    if ! egrep '^backend[[:space:]]*:[[:space:]]*Agg$' ${PGDIR}/matplotlibrc; then
	echo "FLiT reporting will fail without the setting 'backend : Agg' in ${PGDIR}/matplotlibrc.  Please set before using FLiT"
    fi
fi
       
#now we need to add the user and postres to the flit group

sudo addgroup flit
sudo usermod -aG flit sawaya
sudo usermod -aG flit postgres
sudo service postgresql restart
18370matplotlibrc
### MATPLOTLIBRC FORMAT

# This is a sample matplotlib configuration file - you can find a copy
# of it on your system in
# site-packages/matplotlib/mpl-data/matplotlibrc.  If you edit it
# there, please note that it will be overwritten in your next install.
# If you want to keep a permanent local copy that will not be
# overwritten, place it in the following location:
# unix/linux:
#     $HOME/.config/matplotlib/matplotlibrc or
#     $XDG_CONFIG_HOME/matplotlib/matplotlibrc (if $XDG_CONFIG_HOME is set)
# other platforms:
#     $HOME/.matplotlib/matplotlibrc
#
# See http://matplotlib.org/users/customizing.html#the-matplotlibrc-file for
# more details on the paths which are checked for the configuration file.
#
# This file is best viewed in a editor which supports python mode
# syntax highlighting. Blank lines, or lines starting with a comment
# symbol, are ignored, as are trailing comments.  Other lines must
# have the format
#    key : val # optional comment
#
# Colors: for the color values below, you can either use - a
# matplotlib color string, such as r, k, or b - an rgb tuple, such as
# (1.0, 0.5, 0.0) - a hex string, such as ff00ff or #ff00ff - a scalar
# grayscale intensity such as 0.75 - a legal html color name, e.g., red,
# blue, darkslategray

#### CONFIGURATION BEGINS HERE

# The default backend; one of GTK GTKAgg GTKCairo GTK3Agg GTK3Cairo
# CocoaAgg MacOSX Qt4Agg Qt5Agg TkAgg WX WXAgg Agg Cairo GDK PS PDF SVG
# Template.
# You can also deploy your own backend outside of matplotlib by
# referring to the module name (which must be in the PYTHONPATH) as
# 'module://my_backend'.
backend      : Agg

# If you are using the Qt4Agg backend, you can choose here
# to use the PyQt4 bindings or the newer PySide bindings to
# the underlying Qt4 toolkit.
#backend.qt4 : PyQt4        # PyQt4 | PySide

# Note that this can be overridden by the environment variable
# QT_API used by Enthought Tool Suite (ETS); valid values are
# "pyqt" and "pyside".  The "pyqt" setting has the side effect of
# forcing the use of Version 2 API for QString and QVariant.

# The port to use for the web server in the WebAgg backend.
# webagg.port : 8888

# If webagg.port is unavailable, a number of other random ports will
# be tried until one that is available is found.
# webagg.port_retries : 50

# When True, open the webbrowser to the plot that is shown
# webagg.open_in_browser : True

# When True, the figures rendered in the nbagg backend are created with
# a transparent background.
# nbagg.transparent : True

# if you are running pyplot inside a GUI and your backend choice
# conflicts, we will automatically try to find a compatible one for
# you if backend_fallback is True
#backend_fallback: True

#interactive  : False
#toolbar      : toolbar2   # None | toolbar2  ("classic" is deprecated)
#timezone     : UTC        # a pytz timezone string, e.g., US/Central or Europe/Paris

# Where your matplotlib data lives if you installed to a non-default
# location.  This is where the matplotlib fonts, bitmaps, etc reside
#datapath : /home/jdhunter/mpldata


### LINES
# See http://matplotlib.org/api/artist_api.html#module-matplotlib.lines for more
# information on line properties.
#lines.linewidth   : 1.0     # line width in points
#lines.linestyle   : -       # solid line
#lines.color       : blue    # has no affect on plot(); see axes.prop_cycle
#lines.marker      : None    # the default marker
#lines.markeredgewidth  : 0.5     # the line width around the marker symbol
#lines.markersize  : 6            # markersize, in points
#lines.dash_joinstyle : miter        # miter|round|bevel
#lines.dash_capstyle : butt          # butt|round|projecting
#lines.solid_joinstyle : miter       # miter|round|bevel
#lines.solid_capstyle : projecting   # butt|round|projecting
#lines.antialiased : True         # render lines in antialiased (no jaggies)

#markers.fillstyle: full # full|left|right|bottom|top|none

### PATCHES
# Patches are graphical objects that fill 2D space, like polygons or
# circles.  See
# http://matplotlib.org/api/artist_api.html#module-matplotlib.patches
# information on patch properties
#patch.linewidth        : 1.0     # edge width in points
#patch.facecolor        : blue
#patch.edgecolor        : black
#patch.antialiased      : True    # render patches in antialiased (no jaggies)

### FONT
#
# font properties used by text.Text.  See
# http://matplotlib.org/api/font_manager_api.html for more
# information on font properties.  The 6 font properties used for font
# matching are given below with their default values.
#
# The font.family property has five values: 'serif' (e.g., Times),
# 'sans-serif' (e.g., Helvetica), 'cursive' (e.g., Zapf-Chancery),
# 'fantasy' (e.g., Western), and 'monospace' (e.g., Courier).  Each of
# these font families has a default list of font names in decreasing
# order of priority associated with them.  When text.usetex is False,
# font.family may also be one or more concrete font names.
#
# The font.style property has three values: normal (or roman), italic
# or oblique.  The oblique style will be used for italic, if it is not
# present.
#
# The font.variant property has two values: normal or small-caps.  For
# TrueType fonts, which are scalable fonts, small-caps is equivalent
# to using a font size of 'smaller', or about 83% of the current font
# size.
#
# The font.weight property has effectively 13 values: normal, bold,
# bolder, lighter, 100, 200, 300, ..., 900.  Normal is the same as
# 400, and bold is 700.  bolder and lighter are relative values with
# respect to the current weight.
#
# The font.stretch property has 11 values: ultra-condensed,
# extra-condensed, condensed, semi-condensed, normal, semi-expanded,
# expanded, extra-expanded, ultra-expanded, wider, and narrower.  This
# property is not currently implemented.
#
# The font.size property is the default font size for text, given in pts.
# 12pt is the standard value.
#
#font.family         : sans-serif
#font.style          : normal
#font.variant        : normal
#font.weight         : medium
#font.stretch        : normal
# note that font.size controls default text sizes.  To configure
# special text sizes tick labels, axes, labels, title, etc, see the rc
# settings for axes and ticks. Special text sizes can be defined
# relative to font.size, using the following values: xx-small, x-small,
# small, medium, large, x-large, xx-large, larger, or smaller
#font.size           : 12.0
#font.serif          : Bitstream Vera Serif, New Century Schoolbook, Century Schoolbook L, Utopia, ITC Bookman, Bookman, Nimbus Roman No9 L, Times New Roman, Times, Palatino, Charter, serif
#font.sans-serif     : Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif
#font.cursive        : Apple Chancery, Textile, Zapf Chancery, Sand, Script MT, Felipa, cursive
#font.fantasy        : Comic Sans MS, Chicago, Charcoal, Impact, Western, Humor Sans, fantasy
#font.monospace      : Bitstream Vera Sans Mono, Andale Mono, Nimbus Mono L, Courier New, Courier, Fixed, Terminal, monospace

### TEXT
# text properties used by text.Text.  See
# http://matplotlib.org/api/artist_api.html#module-matplotlib.text for more
# information on text properties

#text.color          : black

### LaTeX customizations. See http://wiki.scipy.org/Cookbook/Matplotlib/UsingTex
#text.usetex         : False  # use latex for all text handling. The following fonts
                              # are supported through the usual rc parameter settings:
                              # new century schoolbook, bookman, times, palatino,
                              # zapf chancery, charter, serif, sans-serif, helvetica,
                              # avant garde, courier, monospace, computer modern roman,
                              # computer modern sans serif, computer modern typewriter
                              # If another font is desired which can loaded using the
                              # LaTeX \usepackage command, please inquire at the
                              # matplotlib mailing list
#text.latex.unicode : False # use "ucs" and "inputenc" LaTeX packages for handling
                            # unicode strings.
#text.latex.preamble :  # IMPROPER USE OF THIS FEATURE WILL LEAD TO LATEX FAILURES
                            # AND IS THEREFORE UNSUPPORTED. PLEASE DO NOT ASK FOR HELP
                            # IF THIS FEATURE DOES NOT DO WHAT YOU EXPECT IT TO.
                            # preamble is a comma separated list of LaTeX statements
                            # that are included in the LaTeX document preamble.
                            # An example:
                            # text.latex.preamble : \usepackage{bm},\usepackage{euler}
                            # The following packages are always loaded with usetex, so
                            # beware of package collisions: color, geometry, graphicx,
                            # type1cm, textcomp. Adobe Postscript (PSSNFS) font packages
                            # may also be loaded, depending on your font settings

#text.dvipnghack : None      # some versions of dvipng don't handle alpha
                             # channel properly.  Use True to correct
                             # and flush ~/.matplotlib/tex.cache
                             # before testing and False to force
                             # correction off.  None will try and
                             # guess based on your dvipng version

#text.hinting : auto   # May be one of the following:
                       #   'none': Perform no hinting
                       #   'auto': Use freetype's autohinter
                       #   'native': Use the hinting information in the
                       #             font file, if available, and if your
                       #             freetype library supports it
                       #   'either': Use the native hinting information,
                       #             or the autohinter if none is available.
                       # For backward compatibility, this value may also be
                       # True === 'auto' or False === 'none'.
#text.hinting_factor : 8 # Specifies the amount of softness for hinting in the
                         # horizontal direction.  A value of 1 will hint to full
                         # pixels.  A value of 2 will hint to half pixels etc.

#text.antialiased : True # If True (default), the text will be antialiased.
                         # This only affects the Agg backend.

# The following settings allow you to select the fonts in math mode.
# They map from a TeX font name to a fontconfig font pattern.
# These settings are only used if mathtext.fontset is 'custom'.
# Note that this "custom" mode is unsupported and may go away in the
# future.
#mathtext.cal : cursive
#mathtext.rm  : serif
#mathtext.tt  : monospace
#mathtext.it  : serif:italic
#mathtext.bf  : serif:bold
#mathtext.sf  : sans
#mathtext.fontset : cm # Should be 'cm' (Computer Modern), 'stix',
                       # 'stixsans' or 'custom'
#mathtext.fallback_to_cm : True  # When True, use symbols from the Computer Modern
                                 # fonts when a symbol can not be found in one of
                                 # the custom math fonts.

#mathtext.default : it # The default font to use for math.
                       # Can be any of the LaTeX font names, including
                       # the special name "regular" for the same font
                       # used in regular text.

### AXES
# default face and edge color, default tick sizes,
# default fontsizes for ticklabels, and so on.  See
# http://matplotlib.org/api/axes_api.html#module-matplotlib.axes
#axes.hold           : True    # whether to clear the axes by default on
#axes.facecolor      : white   # axes background color
#axes.edgecolor      : black   # axes edge color
#axes.linewidth      : 1.0     # edge linewidth
#axes.grid           : False   # display grid or not
#axes.titlesize      : large   # fontsize of the axes title
#axes.labelsize      : medium  # fontsize of the x any y labels
#axes.labelpad       : 5.0     # space between label and axis
#axes.labelweight    : normal  # weight of the x and y labels
#axes.labelcolor     : black
#axes.axisbelow      : False   # whether axis gridlines and ticks are below
                               # the axes elements (lines, text, etc)

#axes.formatter.limits : -7, 7 # use scientific notation if log10
                               # of the axis range is smaller than the
                               # first or larger than the second
#axes.formatter.use_locale : False # When True, format tick labels
                                   # according to the user's locale.
                                   # For example, use ',' as a decimal
                                   # separator in the fr_FR locale.
#axes.formatter.use_mathtext : False # When True, use mathtext for scientific
                                     # notation.
#axes.formatter.useoffset      : True    # If True, the tick label formatter
                                         # will default to labeling ticks relative
                                         # to an offset when the data range is very
                                         # small compared to the minimum absolute
                                         # value of the data.

#axes.unicode_minus  : True    # use unicode for the minus symbol
                               # rather than hyphen.  See
                               # http://en.wikipedia.org/wiki/Plus_and_minus_signs#Character_codes
#axes.prop_cycle    : cycler('color', 'bgrcmyk')
                                            # color cycle for plot lines
                                            # as list of string colorspecs:
                                            # single letter, long name, or
                                            # web-style hex
#axes.xmargin        : 0  # x margin.  See `axes.Axes.margins`
#axes.ymargin        : 0  # y margin See `axes.Axes.margins`

#polaraxes.grid      : True    # display grid on polar axes
#axes3d.grid         : True    # display grid on 3d axes

### TICKS
# see http://matplotlib.org/api/axis_api.html#matplotlib.axis.Tick
#xtick.major.size     : 4      # major tick size in points
#xtick.minor.size     : 2      # minor tick size in points
#xtick.major.width    : 0.5    # major tick width in points
#xtick.minor.width    : 0.5    # minor tick width in points
#xtick.major.pad      : 4      # distance to major tick label in points
#xtick.minor.pad      : 4      # distance to the minor tick label in points
#xtick.color          : k      # color of the tick labels
#xtick.labelsize      : medium # fontsize of the tick labels
#xtick.direction      : in     # direction: in, out, or inout

#ytick.major.size     : 4      # major tick size in points
#ytick.minor.size     : 2      # minor tick size in points
#ytick.major.width    : 0.5    # major tick width in points
#ytick.minor.width    : 0.5    # minor tick width in points
#ytick.major.pad      : 4      # distance to major tick label in points
#ytick.minor.pad      : 4      # distance to the minor tick label in points
#ytick.color          : k      # color of the tick labels
#ytick.labelsize      : medium # fontsize of the tick labels
#ytick.direction      : in     # direction: in, out, or inout


### GRIDS
#grid.color       :   black   # grid color
#grid.linestyle   :   :       # dotted
#grid.linewidth   :   0.5     # in points
#grid.alpha       :   1.0     # transparency, between 0.0 and 1.0

### Legend
#legend.fancybox      : False  # if True, use a rounded box for the
                               # legend, else a rectangle
#legend.isaxes        : True
#legend.numpoints     : 2      # the number of points in the legend line
#legend.fontsize      : large
#legend.borderpad     : 0.5    # border whitespace in fontsize units
#legend.markerscale   : 1.0    # the relative size of legend markers vs. original
# the following dimensions are in axes coords
#legend.labelspacing  : 0.5    # the vertical space between the legend entries in fraction of fontsize
#legend.handlelength  : 2.     # the length of the legend lines in fraction of fontsize
#legend.handleheight  : 0.7     # the height of the legend handle in fraction of fontsize
#legend.handletextpad : 0.8    # the space between the legend line and legend text in fraction of fontsize
#legend.borderaxespad : 0.5   # the border between the axes and legend edge in fraction of fontsize
#legend.columnspacing : 2.    # the border between the axes and legend edge in fraction of fontsize
#legend.shadow        : False
#legend.frameon       : True   # whether or not to draw a frame around legend
#legend.framealpha    : None    # opacity of of legend frame
#legend.scatterpoints : 3 # number of scatter points

### FIGURE
# See http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure
#figure.titlesize : medium     # size of the figure title
#figure.titleweight : normal   # weight of the figure title
#figure.figsize   : 8, 6    # figure size in inches
#figure.dpi       : 80      # figure dots per inch
#figure.facecolor : 0.75    # figure facecolor; 0.75 is scalar gray
#figure.edgecolor : white   # figure edgecolor
#figure.autolayout : False  # When True, automatically adjust subplot
                            # parameters to make the plot fit the figure
#figure.max_open_warning : 20  # The maximum number of figures to open through
                               # the pyplot interface before emitting a warning.
                               # If less than one this feature is disabled.

# The figure subplot parameters.  All dimensions are a fraction of the
# figure width or height
#figure.subplot.left    : 0.125  # the left side of the subplots of the figure
#figure.subplot.right   : 0.9    # the right side of the subplots of the figure
#figure.subplot.bottom  : 0.1    # the bottom of the subplots of the figure
#figure.subplot.top     : 0.9    # the top of the subplots of the figure
#figure.subplot.wspace  : 0.2    # the amount of width reserved for blank space between subplots
#figure.subplot.hspace  : 0.2    # the amount of height reserved for white space between subplots

### IMAGES
#image.aspect : equal             # equal | auto | a number
#image.interpolation  : bilinear  # see help(imshow) for options
#image.cmap   : jet               # gray | jet etc...
#image.lut    : 256               # the size of the colormap lookup table
#image.origin : upper             # lower | upper
#image.resample  : False
#image.composite_image : True     # When True, all the images on a set of axes are 
                                  # combined into a single composite image before 
                                  # saving a figure as a vector graphics file, 
                                  # such as a PDF.

### CONTOUR PLOTS
#contour.negative_linestyle : dashed # dashed | solid
#contour.corner_mask        : True   # True | False | legacy

### ERRORBAR PLOTS
#errorbar.capsize : 3             # length of end cap on error bars in pixels

### Agg rendering
### Warning: experimental, 2008/10/10
#agg.path.chunksize : 0           # 0 to disable; values in the range
                                  # 10000 to 100000 can improve speed slightly
                                  # and prevent an Agg rendering failure
                                  # when plotting very large data sets,
                                  # especially if they are very gappy.
                                  # It may cause minor artifacts, though.
                                  # A value of 20000 is probably a good
                                  # starting point.
### SAVING FIGURES
#path.simplify : True   # When True, simplify paths by removing "invisible"
                        # points to reduce file size and increase rendering
                        # speed
#path.simplify_threshold : 0.1  # The threshold of similarity below which
                                # vertices will be removed in the simplification
                                # process
#path.snap : True # When True, rectilinear axis-aligned paths will be snapped to
                  # the nearest pixel when certain criteria are met.  When False,
                  # paths will never be snapped.
#path.sketch : None # May be none, or a 3-tuple of the form (scale, length,
                    # randomness).
                    # *scale* is the amplitude of the wiggle
                    # perpendicular to the line (in pixels).  *length*
                    # is the length of the wiggle along the line (in
                    # pixels).  *randomness* is the factor by which
                    # the length is randomly scaled.

# the default savefig params can be different from the display params
# e.g., you may want a higher resolution, or to make the figure
# background white
#savefig.dpi         : 100      # figure dots per inch
#savefig.facecolor   : white    # figure facecolor when saving
#savefig.edgecolor   : white    # figure edgecolor when saving
#savefig.format      : png      # png, ps, pdf, svg
#savefig.bbox        : standard # 'tight' or 'standard'.
                                # 'tight' is incompatible with pipe-based animation
                                # backends but will workd with temporary file based ones:
                                # e.g. setting animation.writer to ffmpeg will not work,
                                # use ffmpeg_file instead
#savefig.pad_inches  : 0.1      # Padding to be used when bbox is set to 'tight'
#savefig.jpeg_quality: 95       # when a jpeg is saved, the default quality parameter.
#savefig.directory   : ~        # default directory in savefig dialog box,
                                # leave empty to always use current working directory
#savefig.transparent : False    # setting that controls whether figures are saved with a
                                # transparent background by default

# tk backend params
#tk.window_focus   : False    # Maintain shell focus for TkAgg

# ps backend params
#ps.papersize      : letter   # auto, letter, legal, ledger, A0-A10, B0-B10
#ps.useafm         : False    # use of afm fonts, results in small files
#ps.usedistiller   : False    # can be: None, ghostscript or xpdf
                                          # Experimental: may produce smaller files.
                                          # xpdf intended for production of publication quality files,
                                          # but requires ghostscript, xpdf and ps2eps
#ps.distiller.res  : 6000      # dpi
#ps.fonttype       : 3         # Output Type 3 (Type3) or Type 42 (TrueType)

# pdf backend params
#pdf.compression   : 6 # integer from 0 to 9
                       # 0 disables compression (good for debugging)
#pdf.fonttype       : 3         # Output Type 3 (Type3) or Type 42 (TrueType)

# svg backend params
#svg.image_inline : True       # write raster image data directly into the svg file
#svg.image_noscale : False     # suppress scaling of raster data embedded in SVG
#svg.fonttype : 'path'         # How to handle SVG fonts:
#    'none': Assume fonts are installed on the machine where the SVG will be viewed.
#    'path': Embed characters as paths -- supported by most SVG renderers
#    'svgfont': Embed characters as SVG fonts -- supported only by Chrome,
#               Opera and Safari

# docstring params
#docstring.hardcopy = False  # set this when you want to generate hardcopy docstring

# Set the verbose flags.  This controls how much information
# matplotlib gives you at runtime and where it goes.  The verbosity
# levels are: silent, helpful, debug, debug-annoying.  Any level is
# inclusive of all the levels below it.  If your setting is "debug",
# you'll get all the debug and helpful messages.  When submitting
# problems to the mailing-list, please set verbose to "helpful" or "debug"
# and paste the output into your report.
#
# The "fileo" gives the destination for any calls to verbose.report.
# These objects can a filename, or a filehandle like sys.stdout.
#
# You can override the rc default verbosity from the command line by
# giving the flags --verbose-LEVEL where LEVEL is one of the legal
# levels, e.g., --verbose-helpful.
#
# You can access the verbose instance in your code
#   from matplotlib import verbose.
#verbose.level  : silent      # one of silent, helpful, debug, debug-annoying
#verbose.fileo  : sys.stdout  # a log filename, sys.stdout or sys.stderr

# Event keys to interact with figures/plots via keyboard.
# Customize these settings according to your needs.
# Leave the field(s) empty if you don't need a key-map. (i.e., fullscreen : '')

#keymap.fullscreen : f               # toggling
#keymap.home : h, r, home            # home or reset mnemonic
#keymap.back : left, c, backspace    # forward / backward keys to enable
#keymap.forward : right, v           #   left handed quick navigation
#keymap.pan : p                      # pan mnemonic
#keymap.zoom : o                     # zoom mnemonic
#keymap.save : s                     # saving current figure
#keymap.quit : ctrl+w, cmd+w         # close the current figure
#keymap.grid : g                     # switching on/off a grid in current axes
#keymap.yscale : l                   # toggle scaling of y-axes ('log'/'linear')
#keymap.xscale : L, k                # toggle scaling of x-axes ('log'/'linear')
#keymap.all_axes : a                 # enable all axes

# Control location of examples data files
#examples.directory : ''   # directory to look in for custom installation

###ANIMATION settings
#animation.html : 'none'           # How to display the animation as HTML in
                                   # the IPython notebook. 'html5' uses
                                   # HTML5 video tag.
#animation.writer : ffmpeg         # MovieWriter 'backend' to use
#animation.codec : mpeg4           # Codec to use for writing movie
#animation.bitrate: -1             # Controls size/quality tradeoff for movie.
                                   # -1 implies let utility auto-determine
#animation.frame_format: 'png'     # Controls frame format used by temp files
#animation.ffmpeg_path: 'ffmpeg'   # Path to ffmpeg binary. Without full path
                                   # $PATH is searched
#animation.ffmpeg_args: ''         # Additional arguments to pass to ffmpeg
#animation.avconv_path: 'avconv'   # Path to avconv binary. Without full path
                                   # $PATH is searched
#animation.avconv_args: ''         # Additional arguments to pass to avconv
#animation.mencoder_path: 'mencoder'
                                   # Path to mencoder binary. Without full path
                                   # $PATH is searched
#animation.mencoder_args: ''       # Additional arguments to pass to mencoder
#animation.convert_path: 'convert' # Path to ImageMagick's convert binary.
                                   # On Windows use the full path since convert
                                   # is also the name of a system tool.
18370tables.sql
--
-- PostgreSQL database dump
--

-- Dumped from database version 9.5.6
-- Dumped by pg_dump version 9.5.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'SQL_ASCII';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: flit; Type: DATABASE; Schema: -; Owner: -
--

CREATE DATABASE flit WITH TEMPLATE = template0 ENCODING = 'SQL_ASCII' LC_COLLATE = 'C' LC_CTYPE = 'C';


\connect flit

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'SQL_ASCII';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: flit; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON DATABASE flit IS 'The database for collecting all FLiT results';


--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


--
-- Name: plpython3u; Type: PROCEDURAL LANGUAGE; Schema: -; Owner: -
--

CREATE OR REPLACE PROCEDURAL LANGUAGE plpython3u;


SET search_path = public, pg_catalog;

--
-- Name: breakdowntest(text, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION breakdowntest(name text, run integer) RETURNS integer
    LANGUAGE plpython3u
    AS $$

   quer = ("select distinct trunc(score0d, 15) as score, precision, " +
           "compiler, optl, array(select distinct switches from tests " +
           "where name = t1.name and score0 = t1.score0 and precision " +
           "= t1.precision and compiler = t1.compiler and run = t1.run " +
           "and optl = t1.optl)  from tests as t1 where name = '" +
           name + "' and run = " + str(run) + " order by score, compiler")
   res = plpy.execute(quer)
   return res.nrows()
$$;


--
-- Name: cleanupresults(integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION cleanupresults(run integer DEFAULT '-1'::integer) RETURNS integer[]
    LANGUAGE plpython3u
    AS $$
global run
rn = run
if rn == -1:
    r = ("SELECT MAX(index)as index from runs;")
    res = plpy.execute(r)
    rn = res[0]["index"]
    
s = ("update tests set compiler = 'icpc' where compiler ~ " +
     "'.*icpc.*' and run = " + str(rn))
res = plpy.execute(s)
s = ("update tests set host = 'kingspeak' where host ~ " +
     "'.*kingspeak.*' and run = " + str(rn))
res2 = plpy.execute(s)
s = ("update tests set switches=trim(switches)")
res3 = plpy.execute(s)
s = ("update tests set compiler=trim(compiler)")
res4 = plpy.execute(s)
s = ("update tests set compiler='clang++' where compiler='clang++-3.6'")
return [res.nrows(), res2.nrows(), res3.nrows(), res4.nrows()]
$$;


--
-- Name: createschmoo(integer, text[], text[], text[], text, integer, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION createschmoo(run integer, prec text[], compilers text[], optls text[], host text, labsize integer, fname text) RETURNS text
    LANGUAGE plpython3u
    AS $$
from plpy import spiexceptions
from sys import path
from os import environ
path.append('/tmp/flitDbDir/python')
import plotting as pl

host_str = ''
if len(host) > 0:
   host_str = " and host = '" + host + "'"

prec_str = ""
if len(prec) > 0:
   prec_str = " and (precision = '"
   for t in prec:
      prec_str += t + "' or precision = '"
   prec_str = prec_str[:-17] + ")"

optl_str = ""
if len(optls) > 0:
   optl_str = " and (optl = '"
   for t in optls:
      optl_str += t + "' or optl = '"
   optl_str = optl_str[:-12] + ")"

comp_str = ""
if len(compilers) > 0:
   comp_str = " and (compiler = '"
   for c in compilers:
      comp_str += c + "' or compiler = '"
   comp_str = comp_str[:-16] + ")"
   
quer = ("select distinct name from tests as t1 where exists " +
        "(select 1 from tests where t1.name = name and t1.precision " +
        "= precision and t1.score0 != score0 and t1.run = run " +
        "and t1.compiler = compiler and t1.optl = optl and t1.host = host) " +
        "and run = " + str(run) + prec_str + optl_str + comp_str +
        host_str + " order by name")
tests = plpy.execute(quer)

tests_str = ""
if len(tests) > 0:
   tests_str = " and (name = '"
   for t in tests:
      tests_str += t['name'] + "' or name = '"
   tests_str = tests_str[:-12] + ")"

querx = ("select distinct switches, compiler, optl, precision, host " +
        "from tests where " +
        "run = " + str(run) +
        host_str + prec_str + comp_str + optl_str + tests_str +
        " UNION " + 
        "select distinct switches, compiler, optl, precision, host " +
        "from tests where " +
        "run = " + str(run) +
        host_str + prec_str + comp_str + tests_str + " and switches = ''" +
        " and optl = '-O0'" +
        " order by compiler, optl, switches")
x_axis = plpy.execute(querx)
xa_count = len(x_axis)

quer = ("select distinct name from tests where run = " + str(run) +
        prec_str + tests_str + comp_str + " order by name")

y_axis = plpy.execute(quer)
ya_count = len(y_axis)
x_ticks = []
y_ticks = []
z_data = []

x_count = 0
y_count = 0

for x in x_axis:
   x_ticks.append(x['switches'] + ' ' +
                  x['optl'])
   if len(compilers) > 1:
      x_ticks[-1] += ' ' + x['compiler'][0]
for t in y_axis:
   y_ticks.append(t['name'])
   y_count += 1
   quers = ("select distinct score0, switches, compiler, " +
            "optl, host from tests where run = " + str(run) + " and name = '" +
            t['name'] + "'" + prec_str + comp_str + " and optl = '-O0'" +
            host_str + 
            " and switches = '' UNION select distinct score0, switches, " +
            "compiler, optl, host from " +
            " tests where run = " + str(run) +
            " and name = '" + t['name'] + "'" + prec_str + comp_str +
            optl_str + host_str + 
            " order by compiler, optl, switches")
   scores = plpy.execute(quers)
   eq_classes = {}
   line_classes = []
   color = 0
   for x in scores:
      if not x['score0'] in eq_classes:
         eq_classes[x['score0']] = color
         color += 1
   for x in x_axis:
      quer = ("select score0 from tests where name = '" +
              t['name'] + "' and precision = '" + x['precision'] +
              "' and switches = '" + x['switches'] +
              "' and compiler = '" + x['compiler'] +
              "' and optl = '" + x['optl'] + "' and run = " + str(run) +
              " and host = '" + x['host'] + "'")
      score = plpy.execute(quer)
      x_count += 1
      try:
         line_classes.append(eq_classes[score[0]['score0']])
      except KeyError:
         return "key error fetching color: " + quer + " " + quers
   z_data.append(line_classes)

pl.plot(x_ticks, y_ticks, z_data, fname, ', '.join(compilers) +
        ' @ precision(s): ' +
        ', '.join(prec), labsize)

return str(len(z_data))

$$;


--
-- Name: createswitchestable(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION createswitchestable(csv_path text) RETURNS integer
    LANGUAGE plpython3u
    AS $$
from plpy import spiexceptions

count = 0
with open(csv_path) as csv:
   for line in csv:
      vals = line.split(',')
      name = vals[0]
      descr = vals[1]
      quer = ("insert into switch_desc (name, descr) values('" +
              name + "','" + descr + "')")
      plpy.execute(quer)
      count += 1
return count
$$;


--
-- Name: createtimeplot(integer, text[], text[], text[], text, integer, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION createtimeplot(run integer, prec text[], compilers text[], optls text[], host text, labsize integer, fname text) RETURNS text
    LANGUAGE plpython3u
    AS $$
from plpy import spiexceptions
from sys import path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
#import plotting as pl

plt.autoscale(enable=True, axis='both', tight=False)

host_str = ''
if len(host) > 0:
   host_str = " and host = '" + host + "'"

prec_str = ""
if len(prec) > 0:
   prec_str = " and (precision = '"
   for t in prec:
      prec_str += t + "' or precision = '"
   prec_str = prec_str[:-17] + ")"

optl_str = ""
if len(optls) > 0:
   optl_str = " and (optl = '"
   for t in optls:
      optl_str += t + "' or optl = '"
   optl_str = optl_str[:-12] + ")"

comp_str = ""
if len(compilers) > 0:
   comp_str = " and (compiler = '"
   for c in compilers:
      comp_str += c + "' or compiler = '"
   comp_str = comp_str[:-16] + ")"


quer = ("select distinct name from tests where "
        + "run = " + str(run) + prec_str + optl_str + comp_str + host_str
        + " order by name")

tests = plpy.execute(quer)


for t in tests:
   quer = ("select nanosec, score0, switches, optl, compiler, precision from tests where "
           + "run = " + str(run) + prec_str + optl_str + comp_str + host_str
           + " and name = '" + t['name'] + "' order by nanosec")
   x_data = plpy.execute(quer)
   color = 0
   x_axis = []
   colors = {}
   x_labels = []
   values = []
   x_colors = []
   cstrings = ['black', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'red']
   #cmap = cm.get_cmap('Accent')
   colors.clear()
   for x in x_data:
      score = x['score0']
      if not score in colors:
         colors[score] = color
         color += 1
      x_labels.append(x['compiler'] + '_' +
                     x['switches'] + '_' +
                     x['optl'])
      x_colors.append(colors[score])
      values.append(x['nanosec'])
      fig, ax = plt.subplots()
      ax.plot(np.arange(len(x_labels)), values)
      ax.set_xticks([i + .5 for i in range(0, len(x_labels))])
      ax.set_xticklabels(x_labels, rotation=270)
      #ncolor = np.asarray(x_colors) / np.amax(np.asarray(x_colors))
   for xtick, c in zip(ax.get_xticklabels(), x_colors):
      xtick.set_color(cstrings[c])
   ax.tick_params(axis='both', which='major', labelsize=labsize)
   ax.tick_params(axis='both', which='minor', labelsize=labsize)
   plt.tight_layout()
   plt.savefig(fname + '/' + t['name'] + '_' + x['precision'] +
               '_time.pdf')

return str(len(values))

$$;


--
-- Name: dofullflitimport(text, text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION dofullflitimport(path text, notes text) RETURNS integer[]
    LANGUAGE plpython3u
    AS $$
import datetime

query = ("INSERT INTO runs (rdate, notes) "
         "VALUES ('" + str(datetime.datetime.now())  +
         "','" + notes + "')")
plpy.execute(query)
query = ("SELECT MAX(index) from runs")
res = plpy.execute(query)
run = res[0]['max']
query = ("SELECT importflitresults2('" + path + "', " +
         str(run) + ")")
res = plpy.execute(query)
query = ("SELECT importopcoderesults('" + path + "/pins'," +
         str(run) + ")")
res2 = plpy.execute(query)

return [res[0]['importflitresults2'][0],res[0]['importflitresults2'][1],
        res2[0]['importopcoderesults'][0],res2[0]['importopcoderesults'][1]]

$$;


--
-- Name: dumpswitcheslatex(text, text[]); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION dumpswitcheslatex(tex_path text, switches text[]) RETURNS integer
    LANGUAGE plpython3u
    AS $$
from plpy import spiexceptions

count = 0
quer = ("select * from switch_desc")
switchesq = plpy.execute(quer)

with open(tex_path, 'w+') as tp:
    tp.write(' \\begin{tabular}{r|l}\n\tSwitch & Description\\\\ \n\t\\hline\n')
    for sw in switchesq:
        for s in switches:
            if s == sw['name']:
                tp.write('\t' + sw['name'] + ' & ' + sw['descr'].strip() +
		'\\\\ \n')
                count += 1
                break
    tp.write('\\end{tabular}\n')
return count
$$;


--
-- Name: getcurrentuser(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION getcurrentuser() RETURNS text
    LANGUAGE plpython3u
    AS $$
from subprocess import check_output
return check_output('/usr/bin/whoami').decode("utf-8")

$$;


--
-- Name: getpwd(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION getpwd() RETURNS text
    LANGUAGE plpython3u
    AS $$

import os

return os.getcwd()

$$;


--
-- Name: importopcoderesults(text, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION importopcoderesults(path text, run integer) RETURNS integer[]
    LANGUAGE plpython3u
    AS $$
import glob
from plpy import spiexceptions
import os
count = 0
skipped = 0
for f in glob.iglob(path + '/*'):
    fels = os.path.basename(f).split('_')
    if len(fels) != 6:
        continue
    if fels[0] == 'INTEL':
        compiler = 'icpc'
    elif fels[0] == 'GCC':
        compiler = 'g++'
    elif fels[0] == 'CLANG':
        compiler = 'clang++'
    dynamic = False
    host = fels[1]
    flags = fels[2]
    optl = '-' + fels[3]
    precision = fels[4]
    name = fels[5]
    tq = ("SELECT index from tests where " +
          "name = '" + name + "' and " +
          "host = '" + host + "' and " +
          "precision = '" + precision + "' and " +
	  "optl = '" + optl + "' and " +
          "compiler = '" + compiler + "' and " +
          "switches = (select switches from switch_conv where abbrev = '" + flags + "') and " +
          "run = " + str(run))
    res = plpy.execute(tq)
    if res.nrows() != 1:
        dup = res.nrows() > 1
        skq = ("insert into skipped_pin (name, host, precision, optl, " +
               "compiler, switches, run, dup)" +
               " select '" + name + "','" + host + "','" + precision + "','" + 
	       optl + "','" + compiler + "',switch_conv.switches," + str(run) +
	       "," + str(dup) + " from switch_conv where abbrev = '" + flags + "'")
        plpy.execute(skq)
        skipped = skipped + 1
        continue
    tindx = res[0]["index"]
    with open(f) as inf:
        for line in inf:
            l = line.split()
            if len(line.lstrip()) > 0 and line.lstrip()[0] == '#':
                if 'dynamic' in line:
                    dynamic = True
                continue
            if len(l) < 4:
                continue
            opq = ("INSERT INTO opcodes VALUES(" +
                   str(l[0]) + ", '" + l[1] +"')")
            try:
                plpy.execute(opq)
            except spiexceptions.UniqueViolation:
                pass

            cntq = ("INSERT INTO op_counts (test_id, opcode, " +
                    "count, pred_count, dynamic) "+
                    "VALUES(" + str(tindx) + ", " + str(l[0]) +
                    ", " + str(l[2]) + ", " + str(l[3]) + ", " + str(dynamic) + ")")
            plpy.execute(cntq)
            count = count + 1
return [count, skipped]
$$;


--
-- Name: importflitresults(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION importflitresults(path text) RETURNS integer
    LANGUAGE plpython3u
    AS $$
   r = ("SELECT MAX(index)as index from runs;")
   res = plpy.execute(r)
   run = res[0]["index"]
   s = ("COPY tests " +
                "(host, switches, optl, compiler, precision, sort, " +
                "score0d, score0, score1d, score1, name, file) " +
                "FROM '" +
                path +
                "' (DELIMITER ',')")   
   plpy.execute(s)
   s = ("UPDATE tests SET run = " + str(run) + " WHERE run IS NULL;")
   res = plpy.execute(s)
   return res.nrows()
$$;


--
-- Name: importflitresults2(text, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION importflitresults2(path text, run integer) RETURNS integer[]
    LANGUAGE plpython3u
    AS $$
import glob
from plpy import spiexceptions
import os
count = 0
skipped = 0
for f in glob.iglob(path + '/*_out_'):
    with open(f) as inf:
        for line in inf:
            elms = line.split(',')
            host = elms[0].strip()
            switches = elms[1].strip()
            optl = elms[2].strip()
            compiler = elms[3].strip()
            prec = elms[4].strip()
            sort = elms[5].strip()
            score0d = elms[6].strip()
            score0 = elms[7].strip()
            score1d = elms[8].strip()
            score1 = elms[9].strip()
            name = elms[10].strip()
            nseconds = elms[11].strip()
            filen = elms[12].strip()
            quer = ("insert into tests "
                    "(host, switches, optl, compiler, precision, sort, "
                    "score0d, score0, score1d, score1, name, nanosec, file, run) "
                    "VALUES ('" +
                    host + "','" +
                    switches + "','" +
                    optl + "','" +
                    compiler + "','" +
                    prec + "','" +
                    sort + "'," +
                    score0d + ",'" +
                    score0 + "'," +
                    score1d + ",'" +
                    score1 + "','" +
                    name + "'," +
                    nseconds + ",'" +
                    filen + "'," +
                    str(run) + ")")
            try:
                plpy.execute(quer)
            except (spiexceptions.InvalidTextRepresentation,
                    spiexceptions.UndefinedColumn,
		    spiexceptions.NumericValueOutOfRange):
                quer = ("insert into tests "
                        "(host, switches, optl, compiler, precision, sort, "
                        "score0d, score0, score1d, score1, name, nanosec, file, run) "
                        "VALUES ('" +
                        host + "','" +
                        switches + "','" +
                        optl + "','" +
                        compiler + "','" +
                        prec + "','" +
                        sort + "'," +
                        str(0) + ",'" +
                        score0 + "'," +
                        str(0) + ",'" +
                        score1 + "','" +
                        name + "'," +
                        nseconds + ",'" +
                        filen + "'," +
                        str(run) + ")")
                #try:
                plpy.execute(quer)
                #except:
                #    skipped = skipped + 1
                #    continue
            count = count + 1
return [count, skipped]
$$;


--
-- Name: importswitches(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION importswitches(path text) RETURNS integer
    LANGUAGE plpython3u
    AS $$
with open(path) as inf:
    count = 0
    for line in inf:
        spc = line.find(' ')
        if spc == -1:
            abbrev = line
            swts = ''
        else:
            abbrev = line[0:spc]
            swts = line[spc+1:-1]
        q = ("INSERT INTO switch_conv VALUES " +
             "('" + abbrev + "', '" + swts + "')")
        plpy.execute(q)
        count = count + 1
return count
$$;


SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: clusters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE clusters (
    testid integer NOT NULL,
    number integer
);


--
-- Name: op_counts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE op_counts (
    test_id integer NOT NULL,
    opcode integer NOT NULL,
    count integer,
    pred_count integer,
    dynamic boolean NOT NULL
);


--
-- Name: opcodes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE opcodes (
    index integer NOT NULL,
    name text
);


--
-- Name: runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE runs (
    index integer NOT NULL,
    rdate timestamp without time zone,
    notes text
);


--
-- Name: run_index_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE run_index_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: run_index_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE run_index_seq OWNED BY runs.index;


--
-- Name: skipped_pin; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE skipped_pin (
    switches character varying(512),
    "precision" character varying(1),
    sort character varying(2),
    score0 character varying(32),
    score0d numeric(1000,180),
    host character varying(50),
    compiler character varying(50),
    name character varying(255),
    index integer,
    score1 character varying(32),
    score1d numeric(1000,180),
    run integer,
    file character varying(512),
    optl character varying(10),
    dup boolean
);


--
-- Name: switch_conv; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE switch_conv (
    abbrev text NOT NULL,
    switches text
);


--
-- Name: switch_desc; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE switch_desc (
    name character varying(100),
    descr text
);


--
-- Name: tests; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE tests (
    switches character varying(512),
    "precision" character varying(1),
    sort character varying(2),
    score0 character varying(32),
    score0d numeric(1000,180),
    host character varying(50),
    compiler character varying(50),
    name character varying(255),
    index integer NOT NULL,
    score1 character varying(32),
    score1d numeric(1000,180),
    run integer,
    file character varying(512),
    optl character varying(10),
    nanosec numeric(20,0)
);


--
-- Name: tests_colname_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE tests_colname_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: tests_colname_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE tests_colname_seq OWNED BY tests.index;


--
-- Name: index; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY runs ALTER COLUMN index SET DEFAULT nextval('run_index_seq'::regclass);


--
-- Name: index; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY tests ALTER COLUMN index SET DEFAULT nextval('tests_colname_seq'::regclass);


--
-- Name: clusters_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY clusters
    ADD CONSTRAINT clusters_pkey PRIMARY KEY (testid);


--
-- Name: op_counts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY op_counts
    ADD CONSTRAINT op_counts_pkey PRIMARY KEY (test_id, opcode, dynamic);


--
-- Name: opcodes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY opcodes
    ADD CONSTRAINT opcodes_pkey PRIMARY KEY (index);


--
-- Name: runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY runs
    ADD CONSTRAINT runs_pkey PRIMARY KEY (index);


--
-- Name: switch_conv_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY switch_conv
    ADD CONSTRAINT switch_conv_pkey PRIMARY KEY (abbrev);


--
-- Name: switchdesc; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY switch_desc
    ADD CONSTRAINT switchdesc UNIQUE (name);


--
-- Name: tests_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY tests
    ADD CONSTRAINT tests_pkey PRIMARY KEY (index);


--
-- Name: op_counts_opcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY op_counts
    ADD CONSTRAINT op_counts_opcode_fkey FOREIGN KEY (opcode) REFERENCES opcodes(index);


--
-- Name: op_counts_test_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY op_counts
    ADD CONSTRAINT op_counts_test_id_fkey FOREIGN KEY (test_id) REFERENCES tests(index);


--
-- Name: tests_run_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY tests
    ADD CONSTRAINT tests_run_fkey FOREIGN KEY (run) REFERENCES runs(index);


--
-- PostgreSQL database dump complete
--

