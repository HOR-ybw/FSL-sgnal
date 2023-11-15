import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from models.CLIP import *
from utils.DataProcess import *

def  convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def bulit_model(state_config: dict):
    model = CLIP(embed_dim=state_config['embed_dim'],
                     # vision
                     image_resolution=state_config['image_resolution'],
                     vision_layers=state_config['vision_layers'],
                     vision_width=state_config['vision_width'],
                     vision_patch_size=state_config['vision_patch_size'],
                     # text
                     context_length=state_config['context_length'],
                     vocab_size=state_config['vocab_size'],
                     transformer_width=state_config['transformer_width'],  # 512
                     transformer_heads=state_config['transformer_heads'],
                     transformer_layers=state_config['transformer_layers'])
    CLIP.initialize_parameters(model)
    model.to(device)
    if device == "cpu":
        model.float()
    else:
        convert_weights(model)
    return model


def train(epoch, batch_size, image_path, model, optimizer):
    # 加载模型
    #     model, preprocess = load_pretrian_model('ViT-B/16')
    #     print(model)
    #     print(preprocess)
    # 加载数据集
    train_dataloader = load_data(image_path, batch_size)
    #     count=0
    #     设置参数
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt = nn.CrossEntropyLoss().to(device)
#     loss_img = KLLoss()
#     loss_txt = KLLoss()

    #     cos学习率
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0)
    #     model = torch.nn.DataParallel(model).cuda()

    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    #     optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001)

    for i in range(epoch):
        for batch in train_dataloader:
            list_image, list_txt = batch  # list_images is list of image in numpy array(np.uint8), or list of PIL images

            texts = list_txt.to(device)
            images = list_image.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            if device == "cpu":
                ground_truth = torch.arange(batch_size).long().to(device)
            else:
                # ground_truth = torch.arange(batch_size).half().to(device)
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

            # 反向传播
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            optimizer.zero_grad()
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                # convert_models_to_fp32(model)
                optimizer.step()  # 用half训练
                convert_weights(model)
        CosineLR.step()
        print('[%d] loss: %.3f' % (i + 1, total_loss))
    torch.save(model, './my_ViT-B_32v2.0.pt')
    torch.save(model.state_dict(), './my_ViT-B_32v2.0.pth')


def train_fine_tuning(state_config, epoch, batch_size, learning_rate, image_path,
                      param_group=True):
    model = bulit_model(state_config)

    if param_group:
        params_1x = [param for name, param in model.named_parameters()
                     if name not in ["visual.ln_post.weight", "visual.ln_post.bias", "token_embedding.weight",
                                     "token_embedding.bias", "ln_final.weight", "ln_final.bias"]]
        #         params_1x_name = [name for name, param in model.named_parameters()
        #              if name not in ["visual.ln_post.weight", "visual.ln_post.bias","token_embedding.weight",
        #                              "token_embedding.bias","ln_final.weight"]]
        #         print(params_1x_name)
        trainer = torch.optim.SGD([{'params': params_1x,
                                    'lr': learning_rate * 1},
                                   {'params': model.visual.ln_post.parameters(),
                                    'lr': learning_rate * 1},
                                   {'params': model.token_embedding.parameters(),
                                    'lr': learning_rate * 1},
                                   {'params': model.ln_final.parameters(),
                                    'lr': learning_rate * 1}],
                                  lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                  weight_decay=0.0001)

    optimizer = trainer

    train(epoch, batch_size, image_path, model, optimizer)

