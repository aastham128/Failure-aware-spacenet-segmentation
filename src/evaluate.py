model = SimpleUNet().to(device)
model.load_state_dict(torch.load(args.model_path))
model.eval()

val_dataset = SpaceNetDataset(
    image_dir=args.image_dir,
    mask_dir=args.mask_dir,
    limit=args.limit
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

def validate_empty_fp(model, loader, pixel_thresh=50):
    fp_empty = 0
    total_empty = 0

    with torch.no_grad():
        for imgs, masks, is_empty in loader:
            preds = (model(imgs.to(device)) > 0.5).float()

            if is_empty.item():
                total_empty += 1
                if preds.sum().item() > pixel_thresh:
                    fp_empty += 1

    print("Empty-mask FP rate:", fp_empty / max(total_empty, 1))

validate_empty_fp(model, val_loader)

