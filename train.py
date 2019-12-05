import time
import torch

from network import AvatarNet, Encoder
from utils import ImageFolder, imsave, lastest_arverage_value


def network_train(args):
    # set device
    device = torch.device('cuda' if args.gpu_no >= 0 else 'cpu')

    # get network
    network = AvatarNet(args.layers).to(device)

    # get data set
    data_set = ImageFolder(args.content_dir, args.imsize, args.cropsize, args.cencrop)

    # construct a new loss. 
    # only MSE loss of the two features, style and content 
    mse_loss = torch.nn.MSELoss(reduction='mean').to(device)

    #     # get loss calculator
    #     loss_network = Encoder(args.layers).to(device)
    #     mse_loss = torch.nn.MSELoss(reduction='mean').to(device)
    #     loss_seq = {'total':[], 'image':[], 'feature':[], 'tv':[]}
    loss_seq = {'total': [], 'content': [], 'style': []}

    # get optimizer
    # encoder weights are not changing, not updating this one. 
    # also, encoder works as the loss_network, which is originally from perceptual loss 
    for param in network.encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    # training
    for iteration in range(args.max_iter):
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
        input_image = next(iter(data_loader)).to(device)

        # change this line.
        output_image, content_feature, re_content_feature, style_features, re_style_features = network(input_image,
                                                                                                       [input_image],
                                                                                                       train=True)
        # print("input image size", input_image.shape)
        # print("output image size", output_image.shape)
        # print(len(content_feature))
        # print(len(re_content_feature))
        # print("style length", len(style_features[0]))
        # print("re style length",len(re_style_features[0]))
        # print(style_features[0].size)
        # print(re_style_features[0].size)

        # calculate losses, only using the features 
        total_loss = 0

        # similar to the loss network, but different 
        # content feature loss 
        content_loss = 0
        for content_f, re_content_f in zip(content_feature, re_content_feature):
            content_loss += mse_loss(content_f, re_content_f)

        #         content_loss = mse_loss(content_feature, re_content_feature) # for now, list only has one value
        loss_seq['content'].append(content_loss.item())
        total_loss += content_loss  # * args.feature_weight
        # not adding weight for now 

        # style feature loss 
        style_loss = 0
        # print("stlye shape",style_features[0])
        # print("re-stlye shape",re_style_features[0])
        for style_f, re_style_f in zip(style_features[0], re_style_features[0]):
            # print("inside for loop")
            # print("stlye shape", style_f.shape)
            # print("re stlye shape", re_style_f.shape)

            style_loss += mse_loss(style_f, re_style_f)  # these don't match, 44/176
        #         style_loss = mse_loss(style_features[0], re_style_features[0]) # for now, list only has one value
        loss_seq['style'].append(style_loss.item())
        total_loss += style_loss

        #         ## image reconstruction loss
        #         image_loss = mse_loss(output_image, input_image)
        #         loss_seq['image'].append(image_loss.item())
        #         total_loss += image_loss

        #         ## feature reconstruction loss
        #         input_features = loss_network(input_image)
        #         output_features = loss_network(output_image)
        #         feature_loss = 0
        #         for output_feature, input_feature in zip(output_features, input_features):
        #             feature_loss += mse_loss(output_feature, input_feature)
        #         loss_seq['feature'].append(feature_loss.item())
        #         total_loss += feature_loss * args.feature_weight

        #         ## total variation loss
        #         tv_loss = calc_tv_loss(output_image)
        #         loss_seq['tv'].append(tv_loss.item())
        #         total_loss += tv_loss * args.tv_weight

        loss_seq['total'].append(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # print("%s: Iteration: [%d/%d]\tContent Loss: %2.4f\tStyle Loss: %2.4f\tTotal: %2.4f" % (
        #     time.ctime(),
        #     iteration + 1,
        #     args.max_iter,
        #     lastest_arverage_value(loss_seq['content']),
        #     lastest_arverage_value(loss_seq['style']),
        #     lastest_arverage_value(loss_seq['total']))
        #       )

        # print loss log and save network, loss log and output images
        if (iteration + 1) % args.check_iter == 0:
            imsave(torch.cat([input_image, output_image], dim=0), args.save_path + "training_image.png")

            print("%s: Iteration: [%d/%d]\tContent Loss: %2.4f\tStyle Loss: %2.4f\tTotal: %2.4f" % (
            time.ctime(),
            iteration + 1,
            args.max_iter,
            lastest_arverage_value(loss_seq['content']),
            lastest_arverage_value(loss_seq['style']),
            lastest_arverage_value(loss_seq['total']))
                  )
            torch.save({'iteration': iteration + 1,
                        'state_dict': network.state_dict(),
                        'loss_seq': loss_seq},
                       args.save_path + 'check_point_nicolas.pth')

    return network

# we don't use this one
# def calc_tv_loss(x):
#     tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) 
#     tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
#     return tv_loss
