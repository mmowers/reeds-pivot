''' Provide a pivot chart maker example app. Similar to Excel pivot charts,
but with additonal ability to explode into multiple charts.
See README.md for more information.

'''
import bokehpivot as bp
import os
import collections
import bokeh.models.widgets as bmw

dirpath = os.path.dirname(os.path.realpath(__file__))

class ReEDSPivot(bp.BokehPivot):
    def build_top_wdg(self, data_source):
        wdg = collections.OrderedDict()
        wdg['data'] = bmw.TextInput(title='Run(s)', value=data_source, css_classes=['wdgkey-data'])
        wdg['data'].on_change('value', self.update_data)
        return wdg

ReEDSPivot(dirpath)
