from metrice import * 
from DataLoader import *
from virualize import * 
from .models.UnetAttentionGate import UNET3DPPATTEN
from .models.UnetPlusPlus import UNET3DPP
import matplotlib.patches as patches
from DataLoader import BratsDataSet



#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='evaluate', type=str)
    parser.add_argument('--config', default='../evaluate.yaml',type=str)
    args = parser.parse_args()
    return 

def get_dataloader(dataset, path_to_csv, phase, fold = 0, batch_size = 1, num_workers = 4):
    """
    This  function is saving image data in to the list and putting it with torchio.SubjectDataSet
    to split and transform image
    """
    start_time = time.time()
    data = pd.read_csv(path_to_csv)
    train_data = data.loc[data['Fold'] != fold].reset_index(drop = True)
    val_data = data.loc[data['Fold'] == fold].reset_index(drop = True)
    if phase == 'train':
        data = train_data
    else:
        data = val_data
    data_set = dataset(data, phase)
    list_subjects = []
    for i in range(len(data_set)):
        list_subjects.append(data_set[i])
    subject_dataset = tio.SubjectsDataset(list_subjects, transform=get_agumentation(phase))
    data_loader = DataLoader(subject_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=True, 
                            )
    return data_loader

def compute_results(model,
                    dataloader,
                    treshold=0.33):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {"image": [], "GT": [],"Prediction": []}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            imgs, targets =  data['image'][tio.DATA], data['label'][tio.DATA]
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            predictions = (probs >= treshold).float()
            predictions =  predictions.cpu()
            targets = targets.cpu()
            
            
            results["image"].append(imgs.cpu())
            results["GT"].append(targets)
            results["Prediction"].append(predictions)
            
            # only 5 pars
            if (i > 5):    
                return results
        return results

#ensemble learning weights voting 
class Ensemble_models(nn.Module):
    def __init__(self, models, weights=None):
        super(Ensemble_models, self).__init__()
        self.models = models
        self.num_models = len(models)
        self.weights = weights if weights is not None else [1.0/self.num_models]*self.num_models
        
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        avg_output = torch.zeros_like(outputs[0])

        for i, output in enumerate(outputs):
            avg_output += self.weights[i] * output
        
        return avg_output

if __name__ == "__main__":

    args = make_parse()
    cfg = read_yaml(args.config)
    _ = Fold_df_traning(number_split=7)
    #---->update
    cfg.config = args.config
    dataloader = get_dataloader(dataset=BratsDataSet, path_to_csv=cfg.Data.Path, phase='valid', fold=0)
    if cfg.out_channels == 32:
        model = cfg.Model.name(in_channels=4, out_channels=cfg.out_channels , n_classes=3).to(cfg.Device.name)
    else:
        model = cfg.Model.name(in_channels=4, out_channels=32, n_classes=3).to(cfg.Device.name)

    checkpoint = cfg.Model.PathCheckpoint
    model.load_state_dict(torch.load(checkpoint))
    model.eval();
    results = compute_results(model, dataloader, 0.33)

    for img, gt, prediction in zip(results['image'][4:],
                    results['GT'][4:],
                    results['Prediction'][4:]
                    );
    break

    if not os.path.exists('figures'):
        os.makedirs('figures',exist_ok=True)
        
    fig, axes = plt.subplots(1, 3, figsize = (10, 4))

    [ax.axis("off") for ax in axes]
    axes[0].set_title("Image", fontsize=15, weight='bold')
    axes[0].imshow(img_[:,:,50], cmap ='gray')

    axes[1].set_title("Ground Truth", fontsize=15, weight='bold')
    axes[1].imshow(img_[:,:,50], cmap ='gray')
    axes[1].imshow(np.ma.masked_where(wt[:,:,54] == False, wt[:,:,54]),
    cmap='cool_r', alpha=0.8)
    axes[1].imshow(np.ma.masked_where(tc[:,:,54] == False, tc[:,:,54]),
    cmap='cool', alpha=0.7)
    axes[1].imshow(np.ma.masked_where(et[:,:,50] == False, et[:,:,50]),
    cmap='autumn_r', alpha=0.5)
    axes[1].text(10, 105, 'WT', color='w', fontsize=10, weight='bold')
    axes[1].text(25, 105, 'TC', color='w', fontsize=10, weight='bold')
    axes[1].text(40, 105, 'ET', color='w', fontsize=10, weight='bold')
    # Add rectangles as labels
    wt_label = patches.Rectangle((10, 109), 6, 6, linewidth=0, edgecolor='w', facecolor='r')
    tc_label = patches.Rectangle((25, 109), 6, 6, linewidth=0, edgecolor='w', facecolor='g')
    et_label = patches.Rectangle((40, 109), 6, 6, linewidth=0, edgecolor='w', facecolor='y')

    axes[1].add_patch(wt_label)
    axes[1].add_patch(tc_label)
    axes[1].add_patch(et_label)

    axes[2].set_title("Prediction", fontsize=15, weight='bold')
    axes[2].imshow(img_[:,:,50], cmap ='gray')
    axes[2].imshow(np.ma.masked_where(wt_[:,:,54] == False, wt_[:,:,54]),
    cmap='cool_r', alpha=0.8)
    axes[2].imshow(np.ma.masked_where(tc_[:,:,54] == False, tc_[:,:,54]),
    cmap='cool', alpha=0.7)
    axes[2].imshow(np.ma.masked_where(et_[:,:,50] == False, et_[:,:,50]),
    cmap='autumn_r', alpha=0.2)
    axes[2].text(10, 105, 'WT', color='w', fontsize=10, weight='bold')
    axes[2].text(25, 105, 'TC', color='w', fontsize=10, weight='bold')
    axes[2].text(40, 105, 'ET', color='w', fontsize=10, weight='bold')
    # Add rectangles as labels
    wt_label = patches.Rectangle((10, 109), 6, 6, linewidth=0, edgecolor='w', facecolor='r')
    tc_label = patches.Rectangle((25, 109), 6, 6, linewidth=0, edgecolor='w', facecolor='g')
    et_label = patches.Rectangle((40, 109), 6, 6, linewidth=0, edgecolor='w', facecolor='y')

    axes[2].add_patch(wt_label)
    axes[2].add_patch(tc_label)
    axes[2].add_patch(et_label)
    plt.tight_layout()
    plt.savefig('Prediction_Pipeline_B.png')
    plt.show()