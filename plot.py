import csv
from matplotlib import pyplot as plt

results_path = './saved_models/'

def plot_results(file_path=results_path):
    epochs, results_dic = getAllResults(file_path, model_types=['combined','attention'])
    xaxis = range(1, epochs + 2)
    ymin = 0
    ymax = 1

    fig1 = plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.gca().set_title('Training')
    plt.ylabel('Loss')
    plt.plot(xaxis, results_dic['combined']['train_loss'], 'ro-', label='SRN')
    # plt.plot(xaxis, results_dic['attention']['train_loss'], 'go-', label='no Perceptron layer')

    ax = plt.subplot(2, 1, 2)
    plt.gca().set_ylim([ymin, ymax])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(xaxis, results_dic['combined']['train_acc'], 'ro-', label='SRN')
    # plt.plot(xaxis, results_dic['attention']['train_acc'], 'go-', label='no Perceptron layer')

    handles, labels = ax.get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper right')
    fig1.suptitle("Ablation Study of PQ-3H")
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.85)

    fig2 = plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.gca().set_title('Validation')
    plt.ylabel('Loss')
    plt.plot(xaxis, results_dic['combined']['val_loss'], 'ro-', label='SRN')
    # plt.plot(xaxis, results_dic['attention']['val_loss'], 'go-', label='no Perceptron layer')

    ax = plt.subplot(2, 1, 2)
    plt.gca().set_ylim([ymin, ymax])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(xaxis, results_dic['combined']['val_acc'], 'ro-', label='SRN')
    # plt.plot(xaxis, results_dic['attention']['val_acc'], 'go-', label='no Perceptron layer')


    handles, labels = ax.get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper right')
    fig2.suptitle("Ablation Study of PQ-3H")
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.85)
    plt.show()


def getAllResults(file_path, model_types):
    all_results_dic = {}
    epochs = 0
    for model_type in model_types:
        file_name = model_type + '_results.csv'
        try:
            with open(file_path + file_name) as data_file:
                reader = csv.reader(data_file, delimiter=',')
                final_row = ''
                for row in reader:
                    final_row = row
        except FileNotFoundError:
            print('Can\'t find {} results file!'.format(model_type))
            continue
        all_results_dic[model_type] = parseResults(final_row[1:])
        epochs = max(epochs, int(final_row[0].strip()[-1]))
    return epochs, all_results_dic



def parseResults(results):
    results_dic = {}
    result_names = iter(['train_acc','train_loss','val_acc','val_loss'])
    name = next(result_names)
    values = []
    for result in results:
        try:
            result = result.strip()
            if result[-1] == ']':
                values.append(float(result[:-1]))
                results_dic[name] = values
                name = next(result_names)
                values = []
            elif result[0] == '[':
                values.append(float(result[1:]))
            else:
                values.append(float(result))
        except StopIteration:
            break
    return results_dic

if __name__ == '__main__':
    plot_results()