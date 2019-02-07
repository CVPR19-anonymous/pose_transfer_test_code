import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config = config)

import os, logging, shutil, datetime, time, math, pickle
import glob
import argparse
import numpy as np
from tqdm import tqdm, trange
import PIL

import nn
import models
from batches_pg2 import get_batches, plot_batch, postprocess
import batches_deepfashion_mine as bm
import deeploss

N_BOXES = 8


def init_logging(out_base_dir):
    # get unique output directory based on current time
    os.makedirs(out_base_dir, exist_ok = True)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = os.path.join(out_base_dir, now)
    os.makedirs(out_dir, exist_ok = False)
    # copy source code to logging dir to have an idea what the run was about
    this_file = os.path.realpath(__file__)
    assert(this_file.endswith(".py"))
    shutil.copy(this_file, out_dir)
    # copy all py files to logging dir
    src_dir = os.path.dirname(this_file)
    py_files = glob.glob(os.path.join(src_dir, "*.py"))
    for py_file in py_files:
        shutil.copy(py_file, out_dir)
    # init logging
    logging.basicConfig(filename = os.path.join(out_dir, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return out_dir, logger


class Timer(object):
    def __init__(self):
        self.tick()


    def tick(self):
        self.start_time = time.time()


    def tock(self):
        self.end_time = time.time()
        time_since_tick = self.end_time - self.start_time
        self.tick()
        return time_since_tick


class Model(object):
    def __init__(self, opt, out_dir, logger):
        self.batch_size = opt.batch_size
        self.img_shape = 2*[opt.spatial_size] + [3]
        redux = 2
        self.imgn_shape = 2*[opt.spatial_size//(2**redux)] + [N_BOXES*3]
        self.init_batches = opt.init_batches

        self.initial_lr = opt.lr
        self.lr_decay_begin = opt.lr_decay_begin
        self.lr_decay_end = opt.lr_decay_end

        self.out_dir = out_dir
        self.logger = logger
        self.log_frequency = opt.log_freq
        self.ckpt_frequency = opt.ckpt_freq
        self.test_frequency = opt.test_freq
        self.checkpoint_best = False

        self.dropout_p = opt.drop_prob

        self.best_loss = float("inf")
        self.checkpoint_dir = os.path.join(self.out_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok = True)

        self.define_models()
        self.define_graph()


    def define_models(self):
        n_latent_scales = 2
        n_scales = 1 + int(np.round(np.log2(self.img_shape[0]))) - 2
        n_filters = 32
        redux = 2
        self.enc_up_pass = models.make_model(
                "enc_up", models.enc_up,
                n_scales = n_scales - redux,
                n_filters = n_filters*2**redux)
        self.enc_down_pass = models.make_model(
                "enc_down", models.enc_down,
                n_scales = n_scales - redux,
                n_latent_scales = n_latent_scales)
        self.dec_up_pass = models.make_model(
                "dec_up", models.dec_up,
                n_scales = n_scales,
                n_filters = n_filters)
        self.dec_down_pass = models.make_model(
                "dec_down", models.dec_down,
                n_scales = n_scales,
                n_latent_scales = n_latent_scales)
        self.dec_params = models.make_model(
                "dec_params", models.dec_parameters)


    def train_forward_pass(self, x, c, xn, cn, dropout_p, init = False):
        kwargs = {"init": init, "dropout_p": dropout_p}
        # encoder
        hs = self.enc_up_pass(xn, cn, **kwargs)
        es, qs, zs_posterior = self.enc_down_pass(hs, **kwargs)
        # decoder
        gs = self.dec_up_pass(c, **kwargs)
        ds, ps, zs_prior = self.dec_down_pass(gs, zs_posterior, training = True, **kwargs)
        params = self.dec_params(ds[-1], **kwargs)
        activations = hs + es + gs + ds
        return params, qs, ps, activations


    def test_forward_pass(self, c):
        kwargs = {"init": False, "dropout_p": 0.0}
        # decoder
        gs = self.dec_up_pass(c, **kwargs)
        ds, ps, zs_prior = self.dec_down_pass(gs, [], training = False, **kwargs)
        params = self.dec_params(ds[-1], **kwargs)
        return params


    def transfer_pass(self, infer_x, infer_c, generate_c):
        kwargs = {"init": False, "dropout_p": 0.0}
        # infer latent code
        hs = self.enc_up_pass(infer_x, infer_c, **kwargs)
        es, qs, zs_posterior = self.enc_down_pass(hs, **kwargs)
        zs_mean = list(qs)
        # generate from inferred latent code and conditioning
        gs = self.dec_up_pass(generate_c, **kwargs)
        use_mean = True
        if use_mean:
            ds, ps, zs_prior = self.dec_down_pass(gs, zs_mean, training = True, **kwargs)
        else:
            ds, ps, zs_prior = self.dec_down_pass(gs, zs_posterior, training = True, **kwargs)
        params = self.dec_params(ds[-1], **kwargs)
        return params


    def sample(self, params, **kwargs):
        return params


    def likelihood_loss(self, x, params):
        return 5.0*self.vgg19.make_loss_op(x, params)


    def define_graph(self):
        # pretrained net for perceptual loss
        #self.vgg19 = deeploss.JigsawFeatures(session)
        self.vgg19 = deeploss.VGG19Features(session)

        global_step = tf.Variable(0, trainable = False, name = "global_step")
        lr = nn.make_linear_var(
                global_step,
                self.lr_decay_begin, self.lr_decay_end,
                self.initial_lr, 0.0,
                0.0, self.initial_lr)
        kl_weight = nn.make_linear_var(
                global_step,
                self.lr_decay_end // 2, 3 * self.lr_decay_end // 4,
                1e-6, 1.0,
                1e-6, 1.0)
        #kl_weight = tf.to_float(0.1)

        # initialization
        self.x_init = tf.placeholder(
                tf.float32,
                shape = [self.init_batches * self.batch_size] + self.img_shape)
        self.c_init = tf.placeholder(
                tf.float32,
                shape = [self.init_batches * self.batch_size] + self.img_shape)
        self.xn_init = tf.placeholder(
                tf.float32,
                shape = [self.init_batches * self.batch_size] + self.imgn_shape)
        self.cn_init = tf.placeholder(
                tf.float32,
                shape = [self.init_batches * self.batch_size] + self.imgn_shape)
        _ = self.train_forward_pass(
                self.x_init, self.c_init,
                self.xn_init, self.cn_init,
                dropout_p = self.dropout_p, init = True)

        # training
        self.x = tf.placeholder(
                tf.float32,
                shape = [self.batch_size] + self.img_shape)
        self.c = tf.placeholder(
                tf.float32,
                shape = [self.batch_size] + self.img_shape)
        self.xn = tf.placeholder(
                tf.float32,
                shape = [self.batch_size] + self.imgn_shape)
        self.cn = tf.placeholder(
                tf.float32,
                shape = [self.batch_size] + self.imgn_shape)
        # compute parameters of model distribution
        params, qs, ps, activations = self.train_forward_pass(
                self.x, self.c,
                self.xn, self.cn,
                dropout_p = self.dropout_p)
        # sample from model distribution
        sample = self.sample(params)
        # maximize likelihood
        likelihood_loss = self.likelihood_loss(self.x, params)
        kl_loss = tf.to_float(0.0)
        for q, p in zip(qs, ps):
            self.logger.info("Latent shape: {}".format(q.shape.as_list()))
            kl_loss += models.latent_kl(q, p)
        loss = likelihood_loss + kl_weight * kl_loss

        # testing
        test_forward = self.test_forward_pass(self.c)
        test_sample = self.sample(test_forward)

        # reconstruction
        reconstruction_params, _, _, _ = self.train_forward_pass(
                self.x, self.c,
                self.xn, self.cn,
                dropout_p = 0.0)
        self.reconstruction = self.sample(reconstruction_params)

        # optimization
        self.trainable_variables = [v for v in tf.trainable_variables()
                if not v in self.vgg19.variables]
        optimizer = tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5, beta2 = 0.9)
        opt_op = optimizer.minimize(loss, var_list = self.trainable_variables)
        with tf.control_dependencies([opt_op]):
            self.train_op = tf.assign(global_step, global_step + 1)


        # logging and visualization
        self.log_ops = dict()
        self.log_ops["global_step"] = global_step
        self.log_ops["likelihood_loss"] = likelihood_loss
        self.log_ops["kl_loss"] = kl_loss
        self.log_ops["kl_weight"] = kl_weight
        self.log_ops["loss"] = loss
        self.img_ops = dict()
        self.img_ops["sample"] = sample
        self.img_ops["test_sample"] = test_sample
        self.img_ops["x"] = self.x
        self.img_ops["c"] = self.c
        for i in range(N_BOXES):
            self.img_ops["xn{}".format(i)] = self.xn[:,:,:,i*3:(i+1)*3]
        for i, l in enumerate(self.vgg19.losses):
            self.log_ops["vgg_loss_{}".format(i)] = l

        # keep seperate train and validation summaries
        # only training summary contains histograms
        train_summaries = list()
        for k, v in self.log_ops.items():
            train_summaries.append(tf.summary.scalar(k, v))
        self.train_summary_op = tf.summary.merge_all()

        valid_summaries = list()
        for k, v in self.log_ops.items():
            valid_summaries.append(tf.summary.scalar(k+"_valid", v))
        self.valid_summary_op = tf.summary.merge(valid_summaries)

        # all variables for initialization
        self.variables = [v for v in tf.global_variables()
                if not v in self.vgg19.variables]

        self.logger.info("Defined graph")


    def init_graph(self, init_batch):
        self.writer = tf.summary.FileWriter(
                self.out_dir,
                session.graph)
        self.saver = tf.train.Saver(self.variables)
        initializer_op = tf.variables_initializer(self.variables)
        session.run(initializer_op, {
            self.xn_init: init_batch[2],
            self.cn_init: init_batch[3],
            self.x_init: init_batch[0],
            self.c_init: init_batch[1]})
        self.logger.info("Initialized model from scratch")


    def restore_graph(self, restore_path):
        self.writer = tf.summary.FileWriter(
                self.out_dir,
                session.graph)
        self.saver = tf.train.Saver(self.variables)
        self.saver.restore(session, restore_path)
        self.logger.info("Restored model from {}".format(restore_path))


    def fit(self, batches, valid_batches = None):
        start_step = self.log_ops["global_step"].eval(session)
        self.valid_batches = valid_batches
        for batch in trange(start_step, self.lr_decay_end):
            X_batch, C_batch, XN_batch, CN_batch = next(batches)
            feed_dict = {
                    self.xn: XN_batch,
                    self.cn: CN_batch,
                    self.x: X_batch,
                    self.c: C_batch}
            fetch_dict = {"train": self.train_op}
            if self.log_ops["global_step"].eval(session) % self.log_frequency == 0:
                fetch_dict["log"] = self.log_ops
                fetch_dict["img"] = self.img_ops
                fetch_dict["summary"] = self.train_summary_op
            result = session.run(fetch_dict, feed_dict)
            self.log_result(result)


    def log_result(self, result, **kwargs):
        global_step = self.log_ops["global_step"].eval(session)
        if "summary" in result:
            self.writer.add_summary(result["summary"], global_step)
            self.writer.flush()
        if "log" in result:
            for k in sorted(result["log"]):
                v = result["log"][k]
                self.logger.info("{}: {}".format(k, v))
        if "img" in result:
            for k, v in result["img"].items():
                plot_batch(v, os.path.join(
                    self.out_dir,
                    k + "_{:07}.png".format(global_step)))

            if self.valid_batches is not None:
                # validation run
                X_batch, C_batch, XN_batch, CN_batch = next(self.valid_batches)
                feed_dict = {
                        self.xn: XN_batch,
                        self.cn: CN_batch,
                        self.x: X_batch,
                        self.c: C_batch}
                fetch_dict = dict()
                fetch_dict["imgs"] = self.img_ops
                fetch_dict["summary"] = self.valid_summary_op
                fetch_dict["validation_loss"] = self.log_ops["loss"]
                result = session.run(fetch_dict, feed_dict)
                self.writer.add_summary(result["summary"], global_step)
                self.writer.flush()
                # display samples
                imgs = result["imgs"]
                for k, v in imgs.items():
                    plot_batch(v, os.path.join(
                        self.out_dir,
                        "valid_" + k + "_{:07}.png".format(global_step)))
                # log validation loss
                validation_loss = result["validation_loss"]
                self.logger.info("{}: {}".format("validation_loss", validation_loss))
                if self.checkpoint_best and validation_loss < self.best_loss:
                    # checkpoint if validation loss improved
                    self.logger.info("step {}: Validation loss improved from {:.4e} to {:.4e}".format(global_step, self.best_loss, validation_loss))
                    self.best_loss = validation_loss
                    self.make_checkpoint(global_step, prefix = "best_")
        if global_step % self.test_frequency == 0:
            if self.valid_batches is not None:
                # testing
                X_batch, C_batch, XN_batch, CN_batch = next(self.valid_batches)
                x_gen = self.test(C_batch)
                for k in x_gen:
                    plot_batch(x_gen[k], os.path.join(
                        self.out_dir,
                        "testing_{}_{:07}.png".format(k, global_step)))
                # transfer
                bs = X_batch.shape[0]
                imgs = list()
                imgs.append(np.zeros_like(X_batch[0,...]))
                for r in range(bs):
                    imgs.append(C_batch[r,...])
                for i in range(bs):
                    x_infer = XN_batch[i,...]
                    c_infer = CN_batch[i,...]
                    #imgs.append(x_infer)
                    imgs.append(X_batch[i,...])

                    x_infer_batch = x_infer[None,...].repeat(bs, axis = 0)
                    c_infer_batch = c_infer[None,...].repeat(bs, axis = 0)
                    c_generate_batch = C_batch
                    results = model.transfer(x_infer_batch, c_infer_batch, c_generate_batch)
                    for j in range(bs):
                        imgs.append(results[j,...])
                imgs = np.stack(imgs, axis = 0)
                plot_batch(imgs, os.path.join(
                    out_dir,
                    "transfer_{:07}.png".format(global_step)))
        if global_step % self.ckpt_frequency == 0:
            self.make_checkpoint(global_step)


    def make_checkpoint(self, global_step, prefix = ""):
        fname = os.path.join(self.checkpoint_dir, prefix + "model.ckpt")
        self.saver.save(
                session,
                fname,
                global_step = global_step)
        self.logger.info("Saved model to {}".format(fname))


    def test(self, c_batch):
        results = dict()
        results["cond"] = c_batch
        sample = session.run(self.img_ops["test_sample"],
            {self.c: c_batch})
        results["test_sample"] = sample
        return results


    def mcmc(self, c_batch, n_iters = 10):
        results = dict()
        results["cond"] = c_batch
        sample = session.run(
            self.img_ops["test_sample"], {self.c: c_batch})
        results["sample_{}".format(0)] = sample
        for i in range(n_iters - 1):
            sample = session.run(
                    self.img_ops["sample"], {
                        self.x: sample,
                        self.c: c_batch})
            results["sample_{:03}".format(i+1)] = sample
        return results


    def reconstruct(self, x_batch, c_batch):
        return session.run(
                self.reconstruction,
                {self.x: x_batch, self.c: c_batch})


    def transfer(self, x_encode, c_encode, c_decode):
        initialized = getattr(self, "_init_transfer", False)
        if not initialized:
            # transfer
            self.c_generator = tf.placeholder(
                    tf.float32,
                    shape = [self.batch_size] + self.img_shape)
            infer_x = self.xn
            infer_c = self.cn
            generate_c = self.c_generator
            transfer_params = self.transfer_pass(infer_x, infer_c, generate_c)
            self.transfer_mean_sample = self.sample(transfer_params)
            self._init_transfer = True

        return session.run(
                self.transfer_mean_sample, {
                    self.xn: x_encode,
                    self.cn: c_encode,
                    self.c_generator: c_decode})


if __name__ == "__main__":
    default_log_dir = os.path.join(os.getcwd(), "log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", required = True, help = "path to training or testing data index")
    parser.add_argument("--mode", default = "train",
            choices=["train", "test", "mcmc", "add_reconstructions", "transfer"])
    parser.add_argument("--log_dir", default = default_log_dir, help = "path to log into")
    parser.add_argument("--batch_size", default = 8, type = int, help = "batch size")
    parser.add_argument("--init_batches", default = 4, type = int, help = "number of batches for initialization")
    parser.add_argument("--checkpoint", help = "path to checkpoint to restore")
    parser.add_argument("--spatial_size", default = 256, type = int, help = "spatial size to resize images to")
    parser.add_argument("--lr", default = 1e-3, type = float, help = "initial learning rate")
    parser.add_argument("--lr_decay_begin", default = 1000, type = int, help = "steps after which to begin linear lr decay")
    parser.add_argument("--lr_decay_end", default = 100000, type = int, help = "step at which lr is zero, i.e. number of training steps")
    parser.add_argument("--log_freq", default = 250, type = int, help = "frequency to log")
    parser.add_argument("--ckpt_freq", default = 1000, type = int, help = "frequency to checkpoint")
    parser.add_argument("--test_freq", default = 1000, type = int, help = "frequency to test")
    parser.add_argument("--drop_prob", default = 0.1, type = float, help = "Dropout probability")
    parser.add_argument("--mask", dest = "mask", action = "store_true", help = "Use masked data")
    parser.add_argument("--no-mask", dest = "mask", action = "store_false", help = "Do not use mask")
    parser.add_argument("--pairs_path", default='./test_pairs.csv', type=str, help="pairs file path")
    parser.set_defaults(mask = True)
    opt = parser.parse_args()

    if not os.path.exists(opt.data_index):
        raise Exception("Invalid data index: {}".format(opt.data_index))

    out_dir, logger = init_logging(opt.log_dir)
    logger.info(opt)

    if opt.mode == "train":
        batch_size = opt.batch_size
        img_shape = 2*[opt.spatial_size] + [3]
        data_shape = [batch_size] + img_shape
        init_shape = [opt.init_batches * batch_size] + img_shape

        batches = get_batches(data_shape, opt.data_index, mask = opt.mask, train = True)
        init_batches = get_batches(init_shape, opt.data_index, mask = opt.mask, train = True)
        valid_batches = get_batches(data_shape, opt.data_index, mask = opt.mask, train = False)
        logger.info("Number of training samples: {}".format(batches.n))
        logger.info("Number of validation samples: {}".format(valid_batches.n))
        if valid_batches.n == 0:
            valid_batches = None

        model = Model(opt, out_dir, logger)
        if opt.checkpoint is not None:
            model.restore_graph(opt.checkpoint)
        else:
            model.init_graph(next(init_batches))
        model.fit(batches, valid_batches)
    elif opt.mode == "test":
        if not opt.checkpoint:
            raise Exception("Testing requires --checkpoint")
        batch_size = opt.batch_size
        img_shape = 2*[opt.spatial_size] + [3]
        data_shape = [batch_size] + img_shape
        valid_batches = get_batches(data_shape, opt.data_index, mask = opt.mask, train = False)
        model = Model(opt, out_dir, logger)
        model.restore_graph(opt.checkpoint)

        for i in trange(valid_batches.n // batch_size):
            X_batch, C_batch = next(valid_batches)
            x_gen = model.test(C_batch)
            for k in x_gen:
                plot_batch(x_gen[k], os.path.join(
                    out_dir,
                    "testing_{}_{:07}.png".format(k, i)))
    elif opt.mode == "add_reconstructions":
        if not opt.checkpoint:
            raise Exception("Testing requires --checkpoint")
        batch_size = opt.batch_size
        img_shape = 2*[opt.spatial_size] + [3]
        data_shape = [batch_size] + img_shape
        batches = get_batches(data_shape, opt.data_index, mask = opt.mask,
                train = True, return_index_id = True)
        valid_batches = get_batches(data_shape, opt.data_index,
                mask = opt.mask, train = False, return_index_id = True)
        model = Model(opt, out_dir, logger)
        model.restore_graph(opt.checkpoint)

        # open index file to get image filenames and update with
        # reconstruction data
        with open(opt.data_index, "rb") as f:
            index = pickle.load(f)
        index_dir = os.path.dirname(opt.data_index)
        index["reconstruction"] = len(index["imgs"]) * [None]
        index["sample"] = len(index["imgs"]) * [None]

        def process_batches(batches):
            for i in trange(math.ceil(batches.n / batch_size)):
                X_batch, C_batch, I_batch = next(batches)
                # reconstructions
                R_batch = model.reconstruct(X_batch, C_batch)
                R_batch = postprocess(R_batch) # to uint8 for saving
                # samples from pose
                S_batch = model.test(C_batch)["test_sample"]
                S_batch = postprocess(S_batch) # to uint8 for saving
                for batch_i, i in enumerate(I_batch):
                    original_fname = index["imgs"][i]
                    reconstr_fname = original_fname.rsplit(".", 1)[0] + "_reconstruction.png"
                    reconstr_path = os.path.join(index_dir, reconstr_fname)
                    sample_fname = original_fname.rsplit(".", 1)[0] + "_sample.png"
                    sample_path = os.path.join(index_dir, sample_fname)
                    index["reconstruction"][i] = reconstr_path
                    index["sample"][i] = sample_path
                    PIL.Image.fromarray(R_batch[batch_i,...]).save(reconstr_path)
                    PIL.Image.fromarray(S_batch[batch_i,...]).save(sample_path)
        process_batches(batches)
        process_batches(valid_batches)

        # write updated index
        with open(opt.data_index, "wb") as f:
            pickle.dump(index, f)
        logger.info("Wrote {}".format(opt.data_index))

    # elif opt.mode == "transfer":
    #     if not opt.checkpoint:
    #         opt.checkpoint = "log/2017-10-24T16:34:09/checkpoints/model.ckpt-100000"
    #     batch_size = opt.batch_size
    #     img_shape = 2*[opt.spatial_size] + [3]
    #     data_shape = [batch_size] + img_shape
    #     valid_batches = get_batches(data_shape, opt.data_index,
    #             mask = opt.mask, train = False)
    #     model = Model(opt, out_dir, logger)
    #     model.restore_graph(opt.checkpoint)
    #
    #     ids = ["00038", "00281", "01166", "x", "06909", "y", "07586", "07607", "z", "09874"]
    #     for step in trange(10):
    #         # X_batch: images, C_batch: joints, XN_batch: normalized images, CN_batch: normalized joints
    #         X_batch, C_batch, XN_batch, CN_batch = next(valid_batches)
    #         bs = X_batch.shape[0]
    #         imgs = list()
    #         imgs.append(np.zeros_like(X_batch[0,...]))
    #         for r in range(bs):
    #             imgs.append(C_batch[r,...])
    #         for i in range(bs):
    #             x_infer = XN_batch[i,...]
    #             c_infer = CN_batch[i,...]
    #             imgs.append(X_batch[i,...])
    #
    #             x_infer_batch = x_infer[None,...].repeat(bs, axis = 0)
    #             c_infer_batch = c_infer[None,...].repeat(bs, axis = 0)
    #             c_generate_batch = C_batch
    #             results = model.transfer(x_infer_batch, c_infer_batch, c_generate_batch)
    #             for j in range(bs):
    #                 imgs.append(results[j,...])
    #         imgs = np.stack(imgs, axis = 0)
    #         plot_batch(imgs, os.path.join(
    #             out_dir,
    #             "transfer_{}.png".format(ids[step])))

    elif opt.mode == "transfer":
        if not opt.checkpoint:
            opt.checkpoint = "log/2017-10-24T16:34:09/checkpoints/model.ckpt-100000"
        batch_size = opt.batch_size
        img_shape = 2*[opt.spatial_size] + [3]
        data_shape = [batch_size] + img_shape
        valid_batches = bm.get_batches(data_shape, opt.data_index,
                mask = opt.mask, train = False, return_keys=["source_imgs", "source_joints", "source_norm_imgs", "source_norm_joints", "target_imgs", "target_joints", "target_norm_imgs", "target_norm_joints", "from", "to", "target_imgs"], pairs_path=opt.pairs_path)
        model = Model(opt, out_dir, logger)
        model.restore_graph(opt.checkpoint)
        output_img_dir = os.path.join(out_dir, "images")

        # ids = ["00038", "00281", "01166", "x", "06909", "y", "07586", "07607", "z", "09874"]
        for iter in trange(valid_batches.n // batch_size):
            # X_batch: images, C_batch: joints, XN_batch: normalized images, CN_batch: normalized joints
            SX_batch, SC_batch, SXN_batch, SCN_batch, TX_batch, TC_batch, TXN_batch, TCN_batch, from_image, to_image, target_imgs = next(valid_batches)
            results = model.transfer(SXN_batch, SCN_batch, TC_batch)
            bs = SX_batch.shape[0]
            condition_imgs = list()
            target_poses = list()
            final_results = list()
            for i in range(bs):
                target_poses.append(TC_batch[i,...])
            for i in range(bs):
                condition_imgs.append(SX_batch[i,...])
            for i in range(bs):
                final_results.append(results[i,...])
            condition_imgs = np.stack(condition_imgs, axis=0)
            target_poses = np.stack(target_poses, axis=0)
            final_results = np.stack(final_results, axis=0)
            bm.plot_batch_mine(condition_imgs, target_imgs, final_results, output_img_dir, from_image, to_image)
            # bm.plot_normalized_image(XN_batch, from_image, out_dir)
            # bm.plot_normalized_image(CN_batch, to_image, out_dir)


    elif opt.mode == "mcmc":
        if not opt.checkpoint:
            raise Exception("Testing requires --checkpoint")
        batch_size = opt.batch_size
        img_shape = 2*[opt.spatial_size] + [3]
        data_shape = [batch_size] + img_shape
        valid_batches = get_batches(data_shape, opt.data_index, mask = opt.mask, train = False)
        model = Model(opt, out_dir, logger)
        model.restore_graph(opt.checkpoint)

        for i in trange(valid_batches.n // batch_size):
            X_batch, C_batch = next(valid_batches)
            x_gen = model.mcmc(C_batch)
            for k in x_gen:
                plot_batch(x_gen[k], os.path.join(
                    out_dir,
                    "mcmc_{}_{:07}.png".format(k, i)))
    else:
        raise NotImplemented()
