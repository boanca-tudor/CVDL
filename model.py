import torchvision.utils
from torch.nn import MSELoss
from torchvision.datasets.folder import pil_loader

from utils import *
import consts

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits_loss
from torch.optim import Adam
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_layers = nn.ModuleList()

        def add_conv(module_list, name, in_ch, out_ch, kernel, stride, act_fn):
            return module_list.add_module(
                name,
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride
                    ),
                    act_fn
                )
            )

        add_conv(self.conv_layers, 'e_conv_1', in_ch=3, out_ch=64, kernel=5, stride=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_2', in_ch=64, out_ch=128, kernel=5, stride=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_3', in_ch=128, out_ch=256, kernel=5, stride=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_4', in_ch=256, out_ch=512, kernel=5, stride=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_5', in_ch=512, out_ch=1024, kernel=5, stride=2, act_fn=nn.ReLU())

        self.fc_layer = nn.Sequential(
            OrderedDict(
                [
                    ('e_fc_1', nn.Linear(in_features=1024, out_features=consts.NUM_Z_CHANNELS)),
                    ('tanh_1', nn.Tanh())  # normalize to [-1, 1] range
                ]
            )
        )

    def forward(self, face):
        out = face
        for conv_layer in self.conv_layers:
            out = conv_layer(out)
        out = out.flatten(1, -1)
        out = self.fc_layer(out)
        return out


class DiscriminatorZ(nn.Module):
    def __init__(self):
        super(DiscriminatorZ, self).__init__()
        dims = (consts.NUM_Z_CHANNELS, consts.NUM_ENCODER_CHANNELS, consts.NUM_ENCODER_CHANNELS // 2,
                consts.NUM_ENCODER_CHANNELS // 4)
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.layers.add_module(
                'dz_fc_%d' % i,
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU()
                )
            )

        self.layers.add_module(
            'dz_fc_%d' % (i + 1),
            nn.Sequential(
                nn.Linear(out_dim, 1)
            )
        )

    def forward(self, z):
        out = z
        for layer in self.layers:
            out = layer(out)
        return out


class DiscriminatorImg(nn.Module):
    def __init__(self):
        super(DiscriminatorImg, self).__init__()
        in_dims = (3, 16 + consts.LABEL_LEN_EXPANDED, 32, 64)
        out_dims = (16, 32, 64, 128)
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims), 1):
            self.conv_layers.add_module(
                'dimg_conv_%d' % i,
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU()
                )
            )

        self.fc_layers.add_module(
            'dimg_fc_1',
            nn.Sequential(
                nn.Linear(128 * 8 * 8, 1024),
                nn.LeakyReLU()
            )
        )

        self.fc_layers.add_module(
            'dimg_fc_2',
            nn.Sequential(
                nn.Linear(1024, 1)
            )
        )

    def forward(self, imgs, labels, device):
        out = imgs

        for i, conv_layer in enumerate(self.conv_layers, 1):
            out = conv_layer(out)
            if i == 1:
                # concat labels after first conv
                labels_tensor = torch.zeros(torch.Size((out.size(0), labels.size(1), out.size(2), out.size(3))), device=device)
                for img_idx in range(out.size(0)):
                    for label in range(labels.size(1)):
                        labels_tensor[img_idx, label, :, :] = labels[img_idx, label]  # fill a square
                out = torch.cat((out, labels_tensor), 1)

        out = out.flatten(1, -1)
        for fc_layer in self.fc_layers:
            out = fc_layer(out)

        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        mini_size = 4
        self.fc = nn.Sequential(
            nn.Linear(
                consts.NUM_Z_CHANNELS + consts.LABEL_LEN_EXPANDED,
                consts.NUM_GEN_CHANNELS * mini_size ** 2
            ),
            nn.ReLU()
        )

        self.deconv_layers = nn.ModuleList()

        def add_deconv(name, in_dims, out_dims, kernel, stride, act_fn):
            self.deconv_layers.add_module(
                name,
                nn.Sequential(
                    easy_deconv(
                        in_dims=in_dims,
                        out_dims=out_dims,
                        kernel=kernel,
                        stride=stride,
                    ),
                    act_fn
                )
            )

        add_deconv('g_deconv_1', in_dims=(1024, 4, 4), out_dims=(512, 8, 8), kernel=5, stride=2, act_fn=nn.ReLU())
        add_deconv('g_deconv_2', in_dims=(512, 8, 8), out_dims=(256, 16, 16), kernel=5, stride=2, act_fn=nn.ReLU())
        add_deconv('g_deconv_3', in_dims=(256, 16, 16), out_dims=(128, 32, 32), kernel=5, stride=2, act_fn=nn.ReLU())
        add_deconv('g_deconv_4', in_dims=(128, 32, 32), out_dims=(64, 64, 64), kernel=5, stride=2, act_fn=nn.ReLU())
        add_deconv('g_deconv_5', in_dims=(64, 64, 64), out_dims=(32, 128, 128), kernel=5, stride=2, act_fn=nn.ReLU())
        add_deconv('g_deconv_6', in_dims=(32, 128, 128), out_dims=(16, 128, 128), kernel=5, stride=1, act_fn=nn.ReLU())
        add_deconv('g_deconv_7', in_dims=(16, 128, 128), out_dims=(3, 128, 128), kernel=1, stride=1, act_fn=nn.Tanh())

    def _decompress(self, x):
        return x.view(x.size(0), 1024, 4, 4)

    def forward(self, z, age=None, gender=None):
        out = z
        if age is not None and gender is not None:
            label = Label(age, gender) \
                if (isinstance(age, int) and isinstance(gender, int)) \
                else torch.cat((age, gender), 1)
            label = label.repeat(1, 1)
            out = torch.cat((out, label), 1)  # z_l
        out = self.fc(out)
        out = self._decompress(out)
        for i, deconv_layer in enumerate(self.deconv_layers, 1):
            out = deconv_layer(out)
        return out


class Net(object):
    def __init__(self):
        self.E = Encoder()
        self.Dz = DiscriminatorZ()
        self.Dimg = DiscriminatorImg()
        self.G = Generator()

        self.eg_optimizer = Adam(list(self.E.parameters()) + list(self.G.parameters()))
        self.dz_optimizer = Adam(self.Dz.parameters())
        self.di_optimizer = Adam(self.Dimg.parameters())

        self.device = None
        self.cpu()

    def __call__(self, *args, **kwargs):
        self.test(*args, **kwargs)

    def __repr__(self):
        return os.linesep.join([repr(subnet) for subnet in (self.E, self.Dz, self.G)])

    def test(self, image_path, age, gender, target):
        self.eval()
        image_tensor = pil_to_model_tensor_transform(pil_loader(image_path)).to(self.device)

        tensor = image_tensor.repeat(1, 1, 1, 1).to(self.device)
        z = self.E(tensor)

        gender_tensor = -torch.ones(consts.NUM_GENDERS)
        gender_tensor[gender] *= -1
        gender_tensor = gender_tensor.unsqueeze(0)
        gender_tensor = gender_tensor.repeat(1, consts.NUM_AGES // consts.NUM_GENDERS)

        age_tensor = -torch.ones(1, consts.NUM_AGES)
        age_tensor[0][Label.age_transform(age)] *= -1

        l = torch.cat((age_tensor, gender_tensor), 1).to(self.device)
        z_l = torch.cat((z, l), 1)
        generated = self.G(z_l)

        dest = os.path.join(target, 'result.png')
        torchvision.utils.save_image(tensor=generated, fp=dest, nrow=generated.size(0), normalize=True,
                                     range=(-1, 1), padding=4)
        print_timestamp("Saved test result to " + dest)

    def start_training(self, dataset_path, batch_size=64, epochs=50, weight_decay=1e-5, lr=2e-4, should_plot=False,
                       betas=(0.9, 0.999), valid_size=None, where_to_save=None, models_saving='always'):
        where_to_save = where_to_save or default_where_to_save()
        dataset = get_utkface_dataset(dataset_path)
        valid_size = valid_size or batch_size
        valid_dataset, train_dataset = torch.utils.data.random_split(dataset, (valid_size, len(dataset) - valid_size))

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

        input_output_loss = MSELoss()
        nrow = round((2 * batch_size)**0.5)

        for optimizer in (self.eg_optimizer, self.dz_optimizer, self.di_optimizer):
            for param in ('weight_decay', 'betas', 'lr'):
                val = locals()[param]
                if val is not None:
                    optimizer.param_groups[0][param] = val

        loss_tracker = LossTracker(plot=should_plot)
        where_to_save_epoch = ""
        save_count = 0
        paths_for_gif = []

        for epoch in range(1, epochs + 1):
            where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch))
            try:
                if not os.path.exists(where_to_save_epoch):
                    os.makedirs(where_to_save_epoch)
                paths_for_gif.append(where_to_save_epoch)
                losses = defaultdict(lambda: [])
                self.train()  # move to train mode
                for i, (images, labels) in enumerate(train_loader, 1):

                    images = images.to(device=self.device)
                    labels = torch.stack([str_to_tensor(idx_to_class[l], normalize=True) for l in list(labels.numpy())])
                    labels = labels.to(device=self.device)

                    print("DEBUG: iteration: " + str(i) + " images shape: " + str(images.shape))

                    z = self.E(images)

                    # Input\Output Loss
                    z_l = torch.cat((z, labels), 1)
                    generated = self.G(z_l)
                    eg_loss = input_output_loss(generated, images)
                    losses['eg'].append(eg_loss.item())

                    # Total Variance Regularization Loss
                    reg = l1_loss(generated[:, :, :, :-1], generated[:, :, :, 1:]) + l1_loss(generated[:, :, :-1, :], generated[:, :, 1:, :])

                    reg_loss = 0 * reg
                    reg_loss.to(self.device)
                    losses['reg'].append(reg_loss.item())

                    # DiscriminatorZ Loss
                    z_prior = two_sided(torch.rand_like(z, device=self.device))
                    d_z_prior = self.Dz(z_prior)
                    d_z = self.Dz(z)
                    dz_loss_prior = bce_with_logits_loss(d_z_prior, torch.ones_like(d_z_prior))
                    dz_loss = bce_with_logits_loss(d_z, torch.zeros_like(d_z))
                    dz_loss_tot = (dz_loss + dz_loss_prior)
                    losses['dz'].append(dz_loss_tot.item())

                    # Encoder\DiscriminatorZ Loss
                    ez_loss = 0.0001 * bce_with_logits_loss(d_z, torch.ones_like(d_z))
                    ez_loss.to(self.device)
                    losses['ez'].append(ez_loss.item())

                    # DiscriminatorImg Loss
                    d_i_input = self.Dimg(images, labels, self.device)
                    d_i_output = self.Dimg(generated, labels, self.device)

                    di_input_loss = bce_with_logits_loss(d_i_input, torch.ones_like(d_i_input))
                    di_output_loss = bce_with_logits_loss(d_i_output, torch.zeros_like(d_i_output))
                    di_loss_tot = (di_input_loss + di_output_loss)
                    losses['di'].append(di_loss_tot.item())

                    # Generator\DiscriminatorImg Loss
                    dg_loss = 0.0001 * bce_with_logits_loss(d_i_output, torch.ones_like(d_i_output))
                    losses['dg'].append(dg_loss.item())

                    # Start back propagation

                    # Back prop on Encoder\Generator
                    self.eg_optimizer.zero_grad()
                    loss = eg_loss + reg_loss + ez_loss + dg_loss
                    loss.backward(retain_graph=True)
                    self.eg_optimizer.step()

                    # Back prop on DiscriminatorZ
                    self.dz_optimizer.zero_grad()
                    dz_loss_tot.backward(retain_graph=True)
                    self.dz_optimizer.step()

                    # Back prop on DiscriminatorImg
                    self.di_optimizer.zero_grad()
                    di_loss_tot.backward()
                    self.di_optimizer.step()

                    now = datetime.datetime.now()

                logging.info('[{h}:{m}[Epoch {e}] Loss: {t}'.format(h=now.hour, m=now.minute, e=epoch, t=loss.item()))
                print_timestamp(f"[Epoch {epoch:d}] Loss: {loss.item():f}")
                to_save_models = models_saving in ('always', 'tail')
                cp_path = self.save(where_to_save_epoch, to_save_models=to_save_models)
                if models_saving == 'tail':
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, 'losses.png'))

                with torch.no_grad():  # validation
                    self.eval()  # move to eval mode

                    for ii, (images, labels) in enumerate(valid_loader, 1):
                        images = images.to(self.device)
                        labels = torch.stack([str_to_tensor(idx_to_class[label], normalize=True)
                                              for label in list(labels.numpy())])
                        labels = labels.to(self.device)
                        validate_labels = labels.to(self.device)

                        z = self.E(images)
                        z_l = torch.cat((z, validate_labels), 1)
                        generated = self.G(z_l)

                        loss = input_output_loss(images, generated)

                        joined = merge_images(images, generated)

                        file_name = os.path.join(where_to_save_epoch, 'validation.png')
                        torchvision.utils.save_image(tensor=joined, fp=file_name, nrow=nrow, normalize=True,
                                                     range=(-1, 1), padding=4)

                        losses['valid'].append(loss.item())
                        break

                loss_tracker.append_many(**{k: mean(v) for k, v in losses.items()})

                logging.info('[{h}:{m}[Epoch {e}] Loss: {l}'.format(h=now.hour, m=now.minute, e=epoch,
                                                                    l=repr(loss_tracker)))

            except KeyboardInterrupt:
                print_timestamp("{br}CTRL+C detected, saving model{br}".format(br=os.linesep))
                if models_saving != 'never':
                    cp_path = self.save(where_to_save_epoch, to_save_models=True)
                if models_saving == 'tail':
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, 'losses.png'))
                raise

        if models_saving == 'last':
            cp_path = self.save(where_to_save_epoch, to_save_models=True)
        loss_tracker.plot()

    def _mass_fn(self, fn_name, *args, **kwargs):
        """Apply a function to all possible Net's components.

        :return:
        """

        for class_attr in dir(self):
            if not class_attr.startswith('_'):  # ignore private members, for example self.__class__
                class_attr = getattr(self, class_attr)
                if hasattr(class_attr, fn_name):
                    fn = getattr(class_attr, fn_name)
                    fn(*args, **kwargs)

    def to(self, device):
        self._mass_fn('to', device=device)

    def cpu(self):
        self._mass_fn('cpu')
        self.device = torch.device('cpu')

    def cuda(self):
        self._mass_fn('cuda')
        self.device = torch.device('cuda')

    def eval(self):
        """Move Net to evaluation mode.

        :return:
        """
        self._mass_fn('eval')

    def train(self):
        """Move Net to training mode.

        :return:
        """
        self._mass_fn('train')

    def save(self, path, to_save_models=True):
        """Save all state dicts of Net's components.

        :return:
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        if not os.path.isdir(path):
            os.mkdir(path)

        saved = []
        if to_save_models:
            for class_attr_name in dir(self):
                if not class_attr_name.startswith('_'):
                    class_attr = getattr(self, class_attr_name)
                    if hasattr(class_attr, 'state_dict'):
                        state_dict = class_attr.state_dict
                        fname = os.path.join(path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name))
                        torch.save(state_dict, fname)
                        saved.append(class_attr_name)

        if saved:
            print_timestamp("Saved {} to {}".format(', '.join(saved), path))
        elif to_save_models:
            raise FileNotFoundError("Nothing was saved to {}".format(path))
        return path

    def load(self, path, slim=True):
        """Load all state dicts of Net's components.

        :return:
        """
        loaded = []
        for class_attr_name in dir(self):
            if (not class_attr_name.startswith('_')) and ((not slim) or (class_attr_name in ('E', 'G'))):
                class_attr = getattr(self, class_attr_name)
                fname = os.path.join(path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name))
                if hasattr(class_attr, 'load_state_dict') and os.path.exists(fname):
                    class_attr.load_state_dict(torch.load(fname)())
                    loaded.append(class_attr_name)
        if loaded:
            print_timestamp("Loaded {} from {}".format(', '.join(loaded), path))
        else:
            raise FileNotFoundError("Nothing was loaded from {}".format(path))


def create_list_of_img_paths(pattern, start, step):
    result = []
    fname = pattern.format(start)
    while os.path.isfile(fname):
        result.append(fname)
        start += step
        fname = pattern.format(start)
    return result
