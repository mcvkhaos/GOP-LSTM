import datetime,os

def model_selection(Model=None):
    todaydate = str(datetime.datetime.now().date())
    if Model:
        current_model = Model
        print(f'Current model: {current_model} in {os.getcwd()}')
    else:
        current_model = todaydate
        print(f'Current date and model: {current_model} in {os.getcwd()}')

    dirs = os.listdir('./Models/')
    if dirs:
        for d in dirs:
            if d == current_model:
                if os.path.exists(f'./Models/{d}/runs.txt'):
                    with open(f'./Models/{d}/runs.txt','r') as rf:
                        runs = rf.readlines()[0]
                        runs = int(runs)
                        runs += 1
                        print(f'Current number of runs: {runs}')
                    with open(f'./Models/{d}/runs.txt','w') as wf:
                        wf.write(str(runs))
                        wf.write('\n')
                        wf.write(f'Last used: {todaydate}')
                    return f'./Models/{current_model}/'
                else:
                    with open(f'./Models/{d}/runs.txt', 'w') as wf:
                        runs = 0
                        print(f'Current number of runs: {runs}')
                        wf.write(str(runs))
                        wf.write('\n')
                        wf.write(f'Last used: {todaydate}')
                        return f'./Models/{d}/'



    model_name = f'./Models/{current_model}/'
    os.mkdir(model_name)
    with open(f'{model_name}runs.txt', 'w') as wf:
        runs = 0
        print(f'Current number of runs: {runs}')
        wf.write(str(runs))
        wf.write('\n')
        return f'./Models/{current_model}/'


def main():
    model_selection()


if __name__ == "__main__":
    main()
