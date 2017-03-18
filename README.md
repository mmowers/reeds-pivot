# ReEDS Pivot Charts

## Intro
This bokeh app creates pivot charts from ReEDS outputs.

The core of this tool is the same as for the csv pivot tool: https://github.com/mmowers/superpivot. Please see that README for general instructions how to use the pivot functinonality. Additional instructions and features specific to reeds-pivot are descrbed below.

There are two different ways to use this app: On Orion (easiest way), and locally. See the following sections for details on each.

## Running on Orion (easiest)
1. Simply log into Orion and visit http://localhost:5006 in chrome (you probably want to bookmark this URL).
1. Select the app you'd like to run, in this case reeds-pivot (superpivot is the csv pivot chart maker).
1. Go to the *Loading ReEDS data* section below

## Running Locally
1. Follow instructions to install Anaconda for Python 2.7 and Bokeh: https://github.com/mmowers/superpivot#setting-up-from-scratch-if-you-dont-already-have-bokeh
1. This tool reads from the gdx outputs from ReEDS runs. The gams python bindings need to be installed so the necessary python modules are available for reading gdx data into Python. On command line, navigate to the Python API files for Python 2.7, e.g. C:\GAMS\win64\24.7\apifiles\Python\api and run this command:
    ```
    python setup.py install
    ```
1. Finally, git clone this repo to your computer, and on command line (or git bash) enter
    ```
    bokeh serve --show \path\to\this\repo
    ```
    This will launch the bokeh server and a browser window to view the app.
    * Note that I simply used the same command to start the bokeh server process on Orion:
      ```
      bokeh serve D:\CommonGitRepos\Bokeh\reeds-pivot D:\CommonGitRepos\Bokeh\superpivot
      ```
1. Go to the *Loading ReEDS data* section below

## Loading ReEDS data
After starting up the app in a browser window, follow these steps in order to fetch data.
1. *Runs*: In the *Runs* text box, you have three options:
    * Enter a path to a run folder (inside which there is a *gdxfiles/* folder). This works using shared drives too. For example,  *\\\\nrelqnap01d\\ReEDS\\someProject\\runs\\someRun*.
    * Enter a path to a folder containing run folders. For example,  *\\\\nrelqnap01d\\ReEDS\\someProject\\runs*.
    * Enter a path to a csv file that contains a list of runs (see *csv/scenario_template.csv* for an example.)
1. *Filter Scenarios*: A list of scenarios will be fetched after entering a path in *Runs*. Use the *Filter Scenarios* section to reduce the scenarios from which the app will fetch data.
1. *Result*: Select a result from the *Result* select box. It may take a few seconds to fetch the data, depending on the number of scenarios being analyzed.

## ReEDS explorer features
After data is fetched for a given set of runs, a set of dropdowns will appear on the left, allowing you to select axes, series, and explode columns at will for the data (see https://github.com/mmowers/superpivot#running for more info on the core functionality/features). In addition to the core pivot functionality, the following features are specific to the ReEDS explorer:
1. *Presets*: You may select a preset result from the *Preset* select box. For example, for *Generation*, *Stacked Generation* is a preset result.
1. *Meta*: Click the *Meta* section to expand, and see the files used for some default *maps* (to rename and aggregate ReEDS categories), *styles* (to reorder categories and style them), and *merges* (to join more columns, e.g. to add regional aggregations). If you'd like to update any of these files, simply edit the file (only if you're working locally), or point to a new file.
1. *Y-Axis Aggregation*: Select *Sum*, *Average*, or *Weighted Average*. *Weighted Average* requires another field, the *Weighting Factor*. For electricity price, for example, select *load* as the *Weighting Factor*.
1. *Comparisons*: This section allows comparisons across any dimension. You first select the *Operation*, then the column you'd like to *Operate Across*, then the *Base* that you'd like to compare to. Here are a couple examples:
    * Generation differences: Select *Generation* as *Result*, and select *Stacked Gen* under *Presets*. Then, under *Comparisons*, select *Operation*=*Difference*, *Operate Across*=*scenario*, and *Base*=your-base-case.
    * Generation Fraction: Select *Generation* as *Result*, and select *Stacked Gen* under *Presets*. Then, under *Comparisons*, select *Operation*=*Ratio*, *Operate Across*=*tech*, and *Base*=*Total*.
    * Capacity Differences, solve-year-to-solve-year: Select *Capacity* as *Result*, and select *Stacked Capacity* under *Presets*. Then, under *Comparisons*, select *Operation*=*Difference*, *Operate Across*=*year*, and *Base*=*Consecutive*.

## Pro tips
1. Pressing *Alt* will collapse all expandable sections.
1. To suppress the automatic update of the plot while configuring the widgets, simply set *X-axis* to *None* to stop rendering of plots, then make your widget changes, then finally set *X-axis* to the value you'd like.
1. You may interact with the bokeh server with multiple browser windows/tabs, and these interactions will be independent, so you can leave one result open in one tab while you load another in a separate tab, for example.
1. The charts themselves have some useful features shown on the right-hand-side of each chart. For example, hovering over data on a chart will show you the series, x-value, and y-value of the data (not currently working for Area charts). You may also box-zoom or pan (and reset to go back to the initial view). Finally, the charts can be saved as pngs.
1. Save any widget configuration for later use with the *Export Config to URL* button, and copy the resulting URL from the address bar. At a later time, you will be able to load the same view by simply visiting that URL. Note that the bokeh server needs to be running (on the same port) for this to work, and currently you cannot use the same URL on different machines because the paths to files in the *Meta* section are different (work in progress).
1. Download any data you're viewing with the *Download csv* button. It will be downloaded into a timestamped file in the *downloads/* folder.

## Troubleshooting
1. If the app seems to break, simply refresh the page. If a page refresh doesn't work, than restart the bokeh server (if using on local).
1. On Orion, if page refreshes don't work to rectify problems, then notify Matt. He will simply kill the process and restart. If needed on Orion, you may also start a new bokeh server process, but it needs to be on a port that isn't being used (the current apps are running on port 5006, the default port for Bokeh). For example.
    ```
    bokeh serve D:\CommonGitRepos\Bokeh\reeds-pivot --show --port 5007
    ```
