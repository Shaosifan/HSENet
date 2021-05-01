def set_template(args):

    # Set the templates here
    if args.model == 'HSENET':
        args.batch_size = 4
        args.n_basic_modules = 10

    if args.model == 'VDSR' or args.model == 'SRCNN' or args.model == 'LGCNET':
        args.cubic_input = True

    if args.dataset == 'AID':
        args.image_size = 600
    elif args.dataset == 'UCMerced':
        if args.scale[0] == 3:
            args.image_size = 255
        else:
            args.image_size = 256