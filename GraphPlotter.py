from matplotlib import pyplot

class GraphPlotter:

   def __init__(self):
      self.cm = 1 / 2.54  # Centimeters in inches
      self.a4_width = 21 * self.cm # A4 Paper Width
      self.a4_height = 29.7 * self.cm # A4 Paper Height
      self.dpi = 100
      self.facecolor = "white"
      self.edgecolor = "white"
      self.title_fontsize = 30
      self.title_fontweight = "bold"
      self.axis_label_fontsize = 20

   def plot_graph(self, graph_title, dataset, y_label, y_data, x_label, x_data, x_ticks_interval, x_ticks_rotation):
      pyplot.figure(figsize = (self.a4_height, self.a4_width), dpi = self.dpi, facecolor = self.facecolor, edgecolor = self.edgecolor)
      pyplot.title(graph_title, fontsize = self.title_fontsize, fontweight = self.title_fontweight)
      pyplot.ylabel(y_label, fontsize = self.axis_label_fontsize)
      pyplot.plot(y_data)
      pyplot.xlabel(x_label, fontsize = self.axis_label_fontsize)
      pyplot.xticks(range(0, len(dataset.index), x_ticks_interval), x_data.loc[::x_ticks_interval], rotation = x_ticks_rotation)
      pyplot.show()
