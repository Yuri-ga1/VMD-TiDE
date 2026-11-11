import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles

def histPlot(data, info, ROWS, COLUMNS, BINS, filename):
    # Количество строк и столбоц на одной строке

    # Создание файла.
    pdf = PdfPages("fig//"+filename+".pdf")

    # Цикл по страницам
    index = 0

    # Создаем фигуру с несколькими осями.
    figure = plt.figure(figsize=(11, info['height']))

    figure.subplots_adjust(hspace=.3, wspace=.4)
    axes = figure.subplots(ROWS, COLUMNS)

    # Цикл по строкам и столбцам
    for row in range(ROWS):
        for column in range(COLUMNS):

            if len(data[index]) == 3:
                x = data[index][0]
                y = data[index][1]
                z = data[index][2]
                x_w = np.empty(x.shape)
                x_w.fill(1/x.shape[0])
                y_w = np.empty(y.shape)
                y_w.fill(1/y.shape[0])
                z_w = np.empty(z.shape)
                z_w.fill(1/z.shape[0])

                dat = [x, y, z]
                weight_dat = [x_w, y_w, z_w]
                labels = info['labels']

            else:
                x = data[index][0]
                y = data[index][1]
                x_w = np.empty(x.shape)
                x_w.fill(1/x.shape[0])
                y_w = np.empty(y.shape)
                y_w.fill(1/y.shape[0])

                dat = [x, y]
                weight_dat = [x_w, y_w]
                labels = info['labels']

            if ROWS == 1 or COLUMNS == 1:
                axes[index].hist(dat, BINS, weights=weight_dat, label=labels)
                axes[index].legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
                                   fancybox=False, frameon=False, shadow=False, ncol=5)
                axes[index].grid(linestyle='dotted')
                axes[index].set_title(info['titles'][index])
                axes[index].set_xlabel(info['xlabels'][index])
                axes[index].set_ylabel(info['ylabels'][index])
            else:
                axes[row, column].hist(dat, BINS, weights=weight_dat)
                axes[row, column].legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
                                         fancybox=False, frameon=False, shadow=False, ncol=5)
                axes[row, column].grid(linestyle='dotted')
                axes[row, column].set_title(info['titles'][index])
                axes[row, column].set_xlabel(info['xlabels'][index])
                axes[row, column].set_ylabel(info['ylabels'][index])
            index += 1

    # Сохраняем страницу
    pdf.savefig(figure)

    # Сохранение файла
    pdf.close()


def Plot(data, info, ROWS, COLUMNS, filename):
    # Количество строк и столбоц на одной строке

    # Создание файла.
    pdf = PdfPages("fig//"+filename+".pdf")

    # Цикл по страницам
    index = 0

    # Создаем фигуру с несколькими осями.
    figure = plt.figure(figsize=(6, info['height']))
    plt.rcParams['font.size'] = 14
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 1.0

    figure.subplots_adjust(hspace=info['hspace'], wspace=info['wspace'])
    # figure.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.15)
    axes = figure.subplots(ROWS, COLUMNS, sharex=info['sharex'], sharey=info['sharey'])



    if ROWS+COLUMNS == 2:
        for i in range(len(data[index])):
            axes.plot(data[index][i][0], data[index][i][1],
                      info['linestyle'][i], label=info['labels'][i], color=info['colors'][i])
        axes.grid(linestyle='dotted')
        axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
                    fancybox=False, frameon=False, shadow=False, ncol=5)
        axes.set_title(info['titles'][0])
        axes.set_xlabel(info['xlabels'][0])
        axes.set_ylabel(info['ylabels'][0])
        if 'nbins' in info:
            axes.xaxis.set_major_locator(MultipleLocator(base=info['nbins']))

    elif ROWS == 1 or COLUMNS == 1:
        for row in range(ROWS):
            for column in range(COLUMNS):

                for i in range(len(data[index])):
                    axes[index].plot(data[index][i][0], data[index][i]
                                     [1], info['linestyle'][i], label=info['labels'][i], color=info['colors'][i])
                axes[index].grid(linestyle='dotted')
                axes[index].legend(loc='lower center', bbox_to_anchor=(0.5, 1.16),
                                   fancybox=False, frameon=False, shadow=False, ncol=5)
                axes[index].set_title(info['titles'][index])
                axes[index].set_xlabel(info['xlabels'][index])
                axes[index].set_ylabel(info['ylabels'][index])

                if 'nbins' in info:
                    axes[index].xaxis.set_major_locator(MultipleLocator(base=info['nbins']))


                index += 1
    else:
        # Цикл по строкам и столбцам
        for row in range(ROWS):
            for column in range(COLUMNS):
                for i in range(len(data[index])):
                    axes[row, column].plot(
                        data[index][i][0], data[index][i][1], info['linestyle'][i], label=info['labels'][i])
                axes[row, column].grid(linestyle='dotted')
                axes[row, column].legend(loc='best',
                                         fancybox=False, frameon=False, shadow=False, ncol=5)
                axes[row, column].set_title(info['titles'][index])
                axes[row, column].set_xlabel(info['xlabels'][index])
                axes[row, column].set_ylabel(info['ylabels'][index])

                if 'nbins' in info:
                    axes[row, column].xaxis.set_major_locator(MultipleLocator(base=info['nbins']))


                index += 1

    # plt.tight_layout()
    # Сохраняем страницу
    pdf.savefig(figure)

    # Сохранение файла
    pdf.close()


def PlotSingle(data, info, filename):
    # Количество строк и столбоц на одной строке

    # Создание файла.
    pdf = PdfPages("fig//"+filename+".pdf")

    # Цикл по страницам
    index = 0

    # Создаем фигуру с несколькими осями.
    figure = plt.figure(figsize=(11, info['height']))

    for i in range(len(data[index])):
        plt.plot(data[index][i][0], data[index][i][1],
                 info['linestyle'], label=info['labels'][i])
    plt.grid(linestyle='dotted')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=False, frameon=False, shadow=False, ncol=5)
    plt.title(info['titles'][index])
    plt.xlabel(info['xlabels'][index])
    plt.ylabel(info['ylabels'][index])

    # Сохраняем страницу
    pdf.savefig(figure)

    # Сохранение файла
    pdf.close()


def table(data, filename):
    # Количество строк и столбоц на одной строке

    # Создание файла.
    pdf = PdfPages("fig//"+filename+".pdf")

    figure, ax = plt.subplots()

    # hide axes
    figure.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    the_table = ax.table(cellText=data.values, rowLabels=data.index,
                         colLabels=data.columns, loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(7)

    figure.tight_layout()

    # Сохраняем страницу
    pdf.savefig(figure)

    # Сохранение файла
    pdf.close()


def histSingle(data, info, BINS, filename):
    # Количество строк и столбоц на одной строке

    # Создание файла.
    pdf = PdfPages("fig//"+filename+".pdf")

    # Цикл по страницам
    index = 0

    # Создаем фигуру с несколькими осями.
    figure = plt.figure(figsize=(11, info['height']))
    # plt.locator_params(axis='x', nbins=BINS)
    # plt.locator_params(axis='y', nbins=BINS)

    if len(data[index]) == 3:
        x = data[index][0]
        y = data[index][1]
        z = data[index][2]
        x_w = np.empty(x.shape)
        x_w.fill(1/x.shape[0])
        y_w = np.empty(y.shape)
        y_w.fill(1/y.shape[0])
        z_w = np.empty(z.shape)
        z_w.fill(1/z.shape[0])

        dat = [x, y, z]
        weight_dat = [x_w, y_w, z_w]
        labels = info['labels']

    else:
        x = data[index][0]
        y = data[index][1]
        x_w = np.empty(x.shape)
        x_w.fill(1/x.shape[0])
        y_w = np.empty(y.shape)
        y_w.fill(1/y.shape[0])

        dat = [x, y]
        weight_dat = [x_w, y_w]
        labels = info['labels']

    plt.hist(dat, BINS, weights=weight_dat, label=labels)
    plt.legend(loc='upper right')
    plt.grid(linestyle='dotted')
    plt.title(info['titles'][index])
    plt.xlabel(info['xlabels'][index])
    plt.ylabel(info['ylabels'][index])
    # if info['xlim'] in locals():
    # plt.xlim([info['xlim'][0],info['xlim'][1]])

    # Сохраняем страницу
    pdf.savefig(figure)

    # Сохранение файла
    pdf.close()


def hist2Plot(data, info, ROWS, COLUMNS, filename):
    # Количество строк и столбоц на одной строке

    # Создание файла.
    pdf = PdfPages("fig//"+filename+".pdf")

    # Цикл по страницам
    index = 0

    # Создаем фигуру с несколькими осями.
    figure = plt.figure(figsize=(11, info['height']))
    figure.subplots_adjust(hspace=.3, wspace=.4)
    axes = figure.subplots(ROWS, COLUMNS)

    # Цикл по строкам и столбцам
    for row in range(ROWS):
        for column in range(COLUMNS):

            if ROWS == 1 or COLUMNS == 1:
                h = axes[index].hist2d(data[index][0], data[index][1], bins=(
                    info['bins'][0], info['bins'][1]), range=[info['xrange'][index], info['yrange'][index]])
                figure.colorbar(h[3], ax=axes[index])
                # axes[index].legend(loc='upper right')

                axes[index].set_title(info['titles'][index])
                axes[index].set_xlabel(info['xlabels'][index])
                axes[index].set_ylabel(info['ylabels'][index])
                if info['xrange']:
                    axes[index].set_xlim(info['xrange'][index])
                if info['yrange']:
                    axes[index].set_ylim(info['yrange'][index])
            else:
                h = axes[row, column].hist2d(data[index][0], data[index][1], bins=(
                    info['bins'][0], info['bins'][1]), range=[info['xrange'][index], info['yrange'][index]])
                figure.colorbar(h[3], ax=axes[row, column])
                # axes[row, column].legend(loc='upper right')

                axes[row, column].set_title(info['titles'][index])
                axes[row, column].set_xlabel(info['xlabels'][index])
                axes[row, column].set_ylabel(info['ylabels'][index])

                if 'xrange' in info:
                    axes[row, column].set_xlim(info['xrange'][index])
                if 'yrange' in info:
                    axes[row, column].set_ylim(info['yrange'][index])
                # if 'crange' in info:
                #     axes[row, column].set_clim(info['crange'])
            index += 1

    # Сохраняем страницу
    pdf.savefig(figure)

    # Сохранение файла
    pdf.close()


def Venn(data, info, filename):

    # Количество строк и столбоц на одной строке

    # Создание файла.
    pdf = PdfPages("fig//"+filename+".pdf")

    # Цикл по страницам
    index = 0

    # Создаем фигуру с несколькими осями.
    figure = plt.figure(figsize=(11, info['height']))

    venn3(subsets=data, set_labels=info['labels'], alpha=info['alpha'])

    # Сохраняем страницу
    pdf.savefig(figure)

    # Сохранение файла
    pdf.close()
