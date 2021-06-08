from matplotlib import pyplot

class GraphPlotter:

   def __init__(self):
      self.cm = 1 / 2.54  # 1 centimeter in inches
      self.a4_width = 21 * self.cm # A4 paper's width
      self.a4_height = 29.7 * self.cm # A4 paper's height
      self.dpi = 100
      self.facecolor = "white"
      self.edgecolor = "white"
      self.title_fontsize = 30
      self.title_fontweight = "bold"
      self.axis_label_fontsize = 20

   def plotOneCurveGraph(self,
                         graph_title,
                         dataset,
                         y_label,
                         y_data,
                         x_label,
                         x_data,
                         x_ticks_size,
                         x_ticks_rotation):
      pyplot.figure(figsize = (self.a4_height, self.a4_width), dpi = self.dpi, facecolor = self.facecolor, edgecolor = self.edgecolor)
      pyplot.title(graph_title, fontsize = self.title_fontsize, fontweight = self.title_fontweight)
      pyplot.ylabel(y_label, fontsize = self.axis_label_fontsize)
      pyplot.plot(y_data)
      pyplot.xlabel(x_label, fontsize = self.axis_label_fontsize)
      pyplot.xticks(range(0, len(dataset.index), x_ticks_size), x_data.loc[::x_ticks_size], rotation = x_ticks_rotation)
      pyplot.show()

   def plotTwoCurvesGraph(self,
                          graph_title,
                          y_label,
                          first_curve_label,
                          first_curve_color,
                          first_curve_data,
                          second_curve_label,
                          second_curve_color,
                          second_curve_data,
                          x_label,
                          x_ticks_size):
      pyplot.figure(figsize = (self.a4_height, self.a4_width), dpi = self.dpi, facecolor = self.facecolor, edgecolor = self.edgecolor)
      pyplot.plot(range(0, x_ticks_size), first_curve_data, first_curve_color, label = first_curve_label)
      pyplot.plot(range(0, x_ticks_size), second_curve_data, second_curve_color, label = second_curve_label)
      pyplot.title(graph_title, fontsize = self.title_fontsize, fontweight = self.title_fontweight)
      pyplot.ylabel(y_label, fontsize = self.axis_label_fontsize)
      pyplot.xlabel(x_label, fontsize = self.axis_label_fontsize)
      pyplot.legend()
      pyplot.show()

   def plotThreeCurvesGraph(self,
                            graph_title,
                            y_label,
                            first_curve_label,
                            first_curve_color,
                            first_curve_data,
                            second_curve_label,
                            second_curve_color,
                            second_curve_data,
                            third_curve_label,
                            third_curve_color,
                            third_curve_data,
                            x_label,
                            x_ticks_size):
      pyplot.figure(figsize = (self.a4_height, self.a4_width), dpi = self.dpi, facecolor = self.facecolor, edgecolor = self.edgecolor)
      pyplot.plot(range(0, x_ticks_size), first_curve_data, first_curve_color, label = first_curve_label)
      pyplot.plot(range(0, x_ticks_size), second_curve_data, second_curve_color, label = second_curve_label)
      pyplot.plot(range(0, x_ticks_size), third_curve_data, third_curve_color, label = third_curve_label)
      pyplot.title(graph_title, fontsize = self.title_fontsize, fontweight = self.title_fontweight)
      pyplot.ylabel(y_label, fontsize = self.axis_label_fontsize)
      pyplot.xlabel(x_label, fontsize = self.axis_label_fontsize)
      pyplot.legend()
      pyplot.show()
