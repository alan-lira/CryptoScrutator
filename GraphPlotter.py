from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
import math
import numpy

class GraphPlotter:

   def __init__(self):
      self.cm = 1 / 2.54  # 1 centimeter in inches
      self.a4_width = 21 * self.cm # A4 paper's width
      self.a4_height = 29.7 * self.cm # A4 paper's height
      self.dpi = 100
      self.facecolor = "white"
      self.edgecolor = "azure"
      self.title_fontsize = 30
      self.subtitle_fontsize = 12
      self.title_fontweight = "bold"
      self.axis_label_fontsize = 20
      self.xticks_fontweight = "bold"
      self.text_ha = "center"
      self.text_va = "bottom"
      self.text_fontsize = 11
      self.text_fontweight = "bold"

   def getYAxisScaleMagnitude(self, greatest_order_of_magnitude):
      magnitude_initials = None
      scientific_notation_multiplier = None
      if greatest_order_of_magnitude > 3 and greatest_order_of_magnitude <= 6:
         magnitude_initials = "{:1.0f} K"
         scientific_notation_multiplier = 1 * 10 ** - (greatest_order_of_magnitude - (greatest_order_of_magnitude - 3))
      if greatest_order_of_magnitude > 6 and greatest_order_of_magnitude <= 9:
         magnitude_initials = "{:1.0f} M"
         scientific_notation_multiplier = 1 * 10 ** - (greatest_order_of_magnitude - (greatest_order_of_magnitude - 6))
      if greatest_order_of_magnitude > 9 and greatest_order_of_magnitude <= 12:
         magnitude_initials = "{:1.0f} B"
         scientific_notation_multiplier = 1 * 10 ** - (greatest_order_of_magnitude - (greatest_order_of_magnitude - 9))
      if greatest_order_of_magnitude > 12 and greatest_order_of_magnitude <= 15:
         magnitude_initials = "{:1.0f} T"
         scientific_notation_multiplier = 1 * 10 ** - (greatest_order_of_magnitude - (greatest_order_of_magnitude - 12))
      return magnitude_initials, scientific_notation_multiplier

   def plotCurvesGraph(self,
                       graph_title,
                       y_label,
                       x_label,
                       x_ticks_size,
                       curves_labels,
                       curves_colors,
                       curves_datas,
                       greatest_order_of_magnitude):
      pyplot.figure(figsize = (self.a4_height, self.a4_width), dpi = self.dpi, facecolor = self.facecolor, edgecolor = self.edgecolor)
      for index in range(len(curves_labels)):
         pyplot.plot(range(0, x_ticks_size), curves_datas[index], curves_colors[index], label = curves_labels[index])
      pyplot.title(graph_title, fontsize = self.title_fontsize, fontweight = self.title_fontweight)
      pyplot.ylabel(y_label, fontsize = self.axis_label_fontsize)
      pyplot.xlabel(x_label, fontsize = self.axis_label_fontsize)
      pyplot.legend()
      ax = pyplot.gca()
      ax.set_facecolor(self.edgecolor)
      if greatest_order_of_magnitude > 3:
         magnitude_initials, scientific_notation_multiplier = self.getYAxisScaleMagnitude(greatest_order_of_magnitude)
         ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: magnitude_initials.format(int(x * scientific_notation_multiplier))))
      pyplot.show()

   def plotBarsGraph(self,
                     graph_title,
                     graph_subtitle,
                     y_label,
                     x_label,
                     bars_labels,
                     bars_colors,
                     bars_heights,
                     greatest_order_of_magnitude):
      pyplot.figure(figsize = (self.a4_height, self.a4_width), dpi = self.dpi, facecolor = self.facecolor, edgecolor = self.edgecolor)
      x_pos = numpy.arange(len(bars_labels))
      barlist = pyplot.bar(x_pos, bars_heights)
      for bar in barlist:
         yval = 0
         order_of_magnitude = 0
         if bar.get_height() > 0:
            order_of_magnitude = math.floor(math.log10(bar.get_height()))
         if order_of_magnitude <= 0:
            yval = numpy.round(bar.get_height(), -order_of_magnitude)
         else:
            yval = numpy.round(bar.get_height(), 2)
         pyplot.text(bar.get_x() + bar.get_width() / 2., yval, yval, ha = self.text_ha, va = self.text_va, fontsize = self.text_fontsize, fontweight = self.text_fontweight)
      for index in range(len(bars_colors)):
         barlist[index].set_color(bars_colors[index])
      pyplot.suptitle(graph_title, fontsize = self.title_fontsize, fontweight = self.title_fontweight)
      pyplot.title(graph_subtitle, fontsize = self.subtitle_fontsize, fontweight = self.title_fontweight)
      pyplot.ylabel(y_label, fontsize = self.axis_label_fontsize)
      pyplot.xlabel(x_label, fontsize = self.axis_label_fontsize)
      pyplot.xticks(x_pos, bars_labels, fontweight = self.xticks_fontweight)
      ax = pyplot.gca()
      ax.set_facecolor(self.edgecolor)
      if greatest_order_of_magnitude > 3:
         magnitude_initials, scientific_notation_multiplier = self.getYAxisScaleMagnitude(greatest_order_of_magnitude)
         ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: magnitude_initials.format(int(x * scientific_notation_multiplier))))
      pyplot.show()
