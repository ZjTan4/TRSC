import torchvision
import torchvision.transforms as transforms
from models.unet import UNet, UNet_GLCM
import torch
from utils.validate import testval
from custom_dataset import CustomDatsetIO

def main():
    baseline_path = "./model.pt"
    GLCM_path = "./glcm.pt"
    test_dataset = "./data/RUGD"
    img_ext = '.png'

    batch_size = 10
    device = 'cuda:9'
    img_size = (552, 688)
    img_transform = transforms.Compose([
        transforms.Resize(img_size),
        # transforms.Grayscale(num_output_channels=1), 
        # transforms.CenterCrop((544, 688)),
        transforms.ToTensor(), 
    ])
    gt_transform = transforms.Compose([
        transforms.Resize(img_size),
        # transforms.CenterCrop((544, 688)),
        transforms.ToTensor(), 
    ])
    dataset = CustomDatsetIO(test_dataset, img_ext=img_ext, img_transform=img_transform, gt_transform=gt_transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    baseline = UNet(num_of_classes=4).to(device)
    baseline.load_state_dict(torch.load(baseline_path))
    baseline.eval()

    glcm = UNet_GLCM(num_of_classes=4, device=device).to(device)
    glcm.load_state_dict(torch.load(GLCM_path))
    glcm.eval()

    print("Baseline")
    confusion_matrix_baseline = testval(baseline, test_loader, device)
    print(confusion_matrix_baseline)
    print("")
    print("GLCM")
    confusion_matrix_glcm = testval(glcm, test_loader, device)
    print(confusion_matrix_glcm)

if __name__ == "__main__":
    main()