### DO NOT CALL THIS WHEN LATEX IS NOT INSTALLED ###

import matplotlib

def global_settings():
    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "figure.figsize": (9, 5),     # default fig size of 0.9 textwidth
        "figure.dpi": 300,
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 16,               # LaTeX default is 10pt font.
        'axes.titlesize': 15,
        "font.size": 15,
        "legend.fontsize": 16,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
#         "pgf.preamble": [
#             r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
#             r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
#             ]
        }

    matplotlib.rcParams.update(pgf_with_latex)