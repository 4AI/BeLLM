# -*- coding: utf-8 -*-

from angle_emb import *
from billm import MistralForCausalLM, LlamaForCausalLM


class AnglE:
    """
    AnglE. Everything is here👋

    :param model_name_or_path: str, model name or path.
    :param max_length: int. Default 512
    :param model_kwargs: Optional[Dict]. kwargs for model.
    :param lora_config_kwargs: Optional[Dict]. kwargs for peft lora_config.
        details refer to: https://huggingface.co/docs/peft/tutorial/peft_model_config
    :param pooling_strategy: Optional[str]. Pooling strategy.
        Currently support [`cls`, `last`, `avg`, `cls_avg`, `max`]
    :param apply_lora: Optional[bool]. Whether apply lora. Default None.
    :param train_mode: bool. Whether load for training. Default True.
    :param load_kbit: Optional[int]. Specify kbit training from [4, 8, 16]. Default None.
    :param is_llm: Optional[bool]. Whether the model is llm. Default None.
    :param pretrained_model_path: Optional[str]. Default None.
    :param pretrained_lora_path: Optional[str]. Default None.
    :param apply_bfloat16: Optional[bool]. Whether load using torch.bfloat16. Default None.
    :param torch_dtype: Optional[torch.dtype]. Specify torch_dtype. Default None.
    :param device: Optional[str]. Specify device. Default None.
    :param kbit_kwargs: Optional[Dict]. kwargs for kbit. Default None.
        details refer to: https://huggingface.co/docs/peft/package_reference/peft_model#peft.prepare_model_for_kbit_training
    :param **kwargs: Any.
    """  # NOQA
    cfg_file_name = 'angle.config'

    def __init__(self,
                 model_name_or_path: str,
                 max_length: int = 512,
                 model_kwargs: Optional[Dict] = None,
                 lora_config_kwargs: Optional[Dict] = None,
                 pooling_strategy: Optional[str] = None,
                 apply_lora: Optional[bool] = None,
                 train_mode: bool = True,
                 load_kbit: Optional[int] = None,
                 is_llm: Optional[bool] = None,
                 pretrained_model_path: Optional[str] = None,
                 pretrained_lora_path: Optional[str] = None,
                 apply_bfloat16: Optional[bool] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 device: Optional[str] = None,
                 kbit_kwargs: Optional[Dict] = None,
                 **kwargs: Any):
        super().__init__()
        self.max_length = max_length
        self.train_mode = train_mode
        self.pooling_strategy = pooling_strategy
        self.load_kbit = load_kbit
        self.is_llm = is_llm
        if device:
            self.device = device
        else:
            self.device = set_device()
        if is_llm is None:
            self.is_llm = check_llm(model_name_or_path)
            if self.is_llm:
                logger.info('LLM detected, automatically set is_llm=True.'
                            'If it is wrong, you can manually set `is_llm`.')
        self.apply_lora = apply_lora
        if self.apply_lora is None:
            if self.is_llm:
                self.apply_lora = True
                logger.info('LLM detected, automatically set apply_lora=True.'
                            'If it is wrong, you can manually set `apply_lora`.')
        if self.device == 'cuda':
            self.gpu_count = torch.cuda.device_count()
        elif self.device == 'mps':
            self.gpu_count = 1
        else:
            self.gpu_count = 0

        self.prompt = None
        if self.is_llm:
            logger.info('LLM detected, automatically set prompt. '
                        'You can change this setting by manually configuring the `set_prompt()` function.')
            self.set_prompt()

        self.apply_bfloat16 = apply_bfloat16
        if self.apply_bfloat16 is None and 'llama' in model_name_or_path.lower():
            logger.info('LLaMA detected, automatically set `apply_bfloat16=True`. '
                        'You can change this setting by manually configuring the `apply_bfloat16`.')
            self.apply_bfloat16 = True

        if torch_dtype is None:
            torch_dtype = torch.float32 if train_mode else None

        lora_config = None
        if self.apply_lora:
            lora_config = {
                'task_type': TaskType.FEATURE_EXTRACTION,
                'r': 32,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
            }
            if lora_config_kwargs is not None:
                lora_config.update(lora_config_kwargs)
            if train_mode:
                logger.info(f'lora_config={lora_config}')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.is_llm and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        model_kwargs = model_kwargs if model_kwargs is not None else {}
        kbit_kwargs = kbit_kwargs if kbit_kwargs is not None else {}
        if self.is_llm:
            device_map = "auto"
            if 'mistral' in model_name_or_path.lower():
                MODEL_CLASS = MistralForCausalLM
            elif 'llama' in model_name_or_path.lower():
                MODEL_CLASS = LlamaForCausalLM
            else:
                raise ValueError(f'Currently only support Mistral and LlaMa. Unsupported model: {model_name_or_path}')
            if train_mode and self.gpu_count > 1:
                device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            # LLM
            if self.apply_lora:
                lora_config['bias'] = "none"
                lora_config['task_type'] = TaskType.CAUSAL_LM

                if load_kbit in [4, 8]:
                    model = MODEL_CLASS.from_pretrained(
                        model_name_or_path,
                        config=None,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=load_kbit == 4,
                            load_in_8bit=load_kbit == 8,
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False,
                            bnb_4bit_compute_dtype=torch.float32,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type='nf4',
                        ),
                        torch_dtype=torch.float32,
                        device_map=device_map,
                        trust_remote_code=True,
                    )
                    if train_mode:
                        model = prepare_model_for_kbit_training(model, **kbit_kwargs)
                    if pretrained_lora_path is not None:
                        print(f'Load lora weight from {pretrained_lora_path}')
                        model = PeftModel.from_pretrained(
                            model,
                            pretrained_lora_path,
                            torch_dtype=torch.float32,
                            device_map=device_map,
                            is_trainable=train_mode
                        )
                    elif train_mode:
                        if 'target_modules' not in lora_config or lora_config.get('target_modules', None) is None:
                            target_modules = find_all_linear_names(model, linear_type=bnb.nn.Linear4bit if load_kbit == 4 else nn.Linear)
                            lora_config['target_modules'] = target_modules
                            logger.info(f'lora target modules={target_modules}')
                        peft_config = LoraConfig(**lora_config)
                        model = get_peft_model(model, peft_config)
                    model = AnglE.kbit_post_handle(model)
                    self.backbone = model
                else:
                    if self.apply_bfloat16:
                        model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                            output_hidden_states=True,
                                                            trust_remote_code=True).bfloat16()
                    else:
                        model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                            device_map=device_map,
                                                            output_hidden_states=True,
                                                            trust_remote_code=True,
                                                            torch_dtype=torch_dtype or torch.float16)
                    if pretrained_lora_path is not None:
                        print(f'Load lora weight from {pretrained_lora_path}')
                        model = PeftModel.from_pretrained(
                            model,
                            pretrained_lora_path,
                            torch_dtype=torch.float16 if load_kbit == 16 else torch.float32,
                            device_map=device_map,
                            is_trainable=train_mode
                        )
                    else:
                        peft_config = LoraConfig(**lora_config)
                        model = get_peft_model(model, peft_config)

                    self.backbone = model
            else:
                if self.apply_bfloat16:
                    model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                        output_hidden_states=True,
                                                        trust_remote_code=True).bfloat16()
                else:
                    model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                        device_map=device_map,
                                                        output_hidden_states=True,
                                                        trust_remote_code=True,
                                                        load_in_8bit=load_kbit == 8,
                                                        torch_dtype=torch_dtype or torch.float16)
                self.backbone = model
        else:
            # non-LLMs
            if self.apply_lora:
                model = AutoModel.from_pretrained(pretrained_model_path or model_name_or_path, trust_remote_code=True)
                if pretrained_lora_path is not None:
                    model = PeftModel.from_pretrained(
                        model,
                        pretrained_lora_path,
                        is_trainable=train_mode
                    )
                else:
                    if 'target_modules' not in lora_config or lora_config.get('target_modules', None) is None:
                        target_modules = find_all_linear_names(model)
                        lora_config['target_modules'] = target_modules
                        logger.info(f'lora target modules={target_modules}')
                    peft_config = LoraConfig(**lora_config)
                    model = get_peft_model(model, peft_config)
                self.backbone = model
            else:
                if pretrained_model_path is not None:
                    logger.info(f'Load pretrained model from {pretrained_model_path}')
                self.backbone = AutoModel.from_pretrained(
                    pretrained_model_path or model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype or "auto")

        if train_mode and self.apply_lora:
            self.backbone.print_trainable_parameters()

        self.backbone.config.use_cache = False
        self.pooler = Pooler(
            self.backbone,
            pooling_strategy=self.pooling_strategy,
            padding_strategy=self.tokenizer.padding_side,
            is_llm=self.is_llm)

        # full_backbone is used to 2DMSE inference
        self.full_backbone = None
        self.__cfg = {
            'model_name_or_path': model_name_or_path,
            'max_length': max_length,
            'model_kwargs': model_kwargs,
            'pooling_strategy': pooling_strategy,
            'lora_config_kwargs': lora_config,
            'apply_lora': apply_lora,
        }
        self.__cfg.update(kwargs)

    def cuda(self):
        if self.load_kbit is None:
            self.backbone = self.backbone.to(torch.device(self.device))
        return self

    def to(self, device: Any):
        if isinstance(device, str):
            device = torch.device(device)
        self.backbone = self.backbone.to(device)
        self.device = device
        return self

    @staticmethod
    def kbit_post_handle(model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.float32)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    module = module.to(torch.float32)
        return model

    @staticmethod
    def find_pth_path(dirpath: str, config: Dict) -> str:
        if config['save_mode'] == 'best':
            return os.path.join(dirpath, config['best_file_name'])

        pth_list = []
        for fname in os.listdir(dirpath):
            if fname.endswith('.pth'):
                epoch = int(re.search(r'\d+', fname).group())
                pth_list.append((epoch, fname))
        pth_list = sorted(pth_list, key=lambda x: x[0], reverse=True)
        return os.path.join(dirpath, pth_list[0][1])

    @staticmethod
    def from_pretrained(model_name_or_path: str,
                        pretrained_model_path: Optional[str] = None,
                        pretrained_lora_path: Optional[str] = None,
                        is_llm: Optional[bool] = None,
                        pooling_strategy: str = 'cls',
                        train_mode: bool = False,
                        **kwargs):
        """
        Load AnglE from pretrained model.

        :param model_name_or_path: str, model name or path. Required.
        :param pretrained_model_path: Optional[str].
        :param pretrained_lora_path: Optional[str].
        :param is_llm: Optional[bool].
        :param pooling_strategy: str. Pooling Strategy. Default `cls`.
        :param train_mode: bool. Default False.
        :param kwargs: Other kwargs for AnglE.

        :return: AnglE object.

        Example::

                from angle_emb import AnglE

                angle = AnglE.from_pretrained(model_name_or_path)
                # fit
                angle.fit(*args, **kwargs)
                # inference
                angle.encode(*args, **kwargs)
        """
        angle = AnglE(model_name_or_path,
                      is_llm=is_llm,
                      pretrained_model_path=pretrained_model_path,
                      pretrained_lora_path=pretrained_lora_path,
                      pooling_strategy=pooling_strategy,
                      train_mode=train_mode,
                      **kwargs)
        return angle

    @staticmethod
    def load_config(fpath: str) -> Dict:
        with open(fpath, 'r', encoding='utf-8') as reader:
            return json.load(reader)

    def save_config(self, fpath: str):
        with open(fpath, 'w', encoding='utf-8') as writer:
            json.dump(self.__cfg, writer, ensure_ascii=False, indent=2)

    def detect_dataset_format(self, ds: Dataset):
        for obj in ds:
            return obj['extra']['dataset_format']

    def fit(self,
            train_ds: Dataset,
            valid_ds: Optional[Dataset] = None,
            batch_size: int = 32,
            output_dir: Optional[str] = None,
            epochs: int = 1,
            learning_rate: float = 1e-5,
            warmup_steps: int = 1000,
            logging_steps: int = 10,
            eval_steps: Optional[int] = None,
            save_steps: int = 100,
            save_strategy: str = 'steps',
            save_total_limit: int = 10,
            gradient_accumulation_steps: int = 1,
            fp16: Optional[bool] = None,
            argument_kwargs: Optional[Dict] = None,
            trainer_kwargs: Optional[Dict] = None,
            loss_kwargs: Optional[Dict] = None,
            apply_tdmse: bool = False,
            filter_duplicate: bool = True):
        """
        Fit using AnglE.

        :param train_ds: Dataset. tokenized train dataset. Required.
        :param valid_ds: Optional[Dataset]. tokenized valid dataset. Default None.
        :param batch_size: int. Default 32.
        :param output_dir: Optional[str]. save dir. Default None.
        :param epochs: int. Default 1.
        :param learning_rate: float. Default 1e-5.
        :param warmup_steps: int. Default 1000.
        :param logging_steps: int. Default 10.
        :param eval_steps: Optional[int]. Default None.
        :param save_steps: int. Default 100.
        :param save_strategy: str. Default steps.
        :param save_total_limit: int. Default 10.
        :param gradient_accumulation_steps: int. Default 1.
        :param fp16: Optional[bool]. Default None.
        :param argument_kwargs: Optional[Dict]. kwargs for TrainingArguments.
            refer to: https://huggingface.co/docs/transformers/v4.37.0/en/main_classes/trainer#transformers.TrainingArguments
        :param trainer_kwargs: Optional[Dict]. kwargs for AngleTrainer.
        :param loss_kwargs: Optional[Dict]. kwargs for AngleLoss.
        :param apply_tdmse: bool, whether apply TDMSE training.
        """  # NOQA
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        # save config
        self.save_config(os.path.join(output_dir, AnglE.cfg_file_name))

        if self.gpu_count > 1:
            gradient_accumulation_steps = gradient_accumulation_steps // self.gpu_count
        if fp16 is None and self.is_llm:
            fp16 = True
        else:
            fp16 = False
        if argument_kwargs is None:
            argument_kwargs = {}
        if trainer_kwargs is None:
            trainer_kwargs = {}
        callbacks = None
        if valid_ds is not None:
            best_ckpt_dir = None
            if output_dir is not None:
                best_ckpt_dir = os.path.join(output_dir, 'best-checkpoint')
            evaluate_callback = EvaluateCallback(self.backbone, valid_ds,
                                                 partial(self.evaluate, batch_size=batch_size, device=self.device),
                                                 save_dir=best_ckpt_dir)
            callbacks = [evaluate_callback]

        CustomTrainer = AngleTDMSETrainer if apply_tdmse else AngleTrainer
        trainer = CustomTrainer(
            pooler=self.pooler,
            model=self.backbone,
            dataset_format=self.detect_dataset_format(train_ds),
            train_dataset=train_ds,
            eval_dataset=None,
            loss_kwargs=loss_kwargs,
            tokenizer=self.tokenizer,
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                fp16=fp16,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                eval_steps=eval_steps,
                save_steps=save_steps,
                output_dir=output_dir,
                save_total_limit=save_total_limit,
                load_best_model_at_end=False,
                ddp_find_unused_parameters=False if self.gpu_count > 1 else None,
                label_names=['labels', 'seperate_ids', 'extra'],
                **argument_kwargs,
            ),
            callbacks=callbacks,
            data_collator=AngleDataCollator(
                self.tokenizer, return_tensors="pt", max_length=self.max_length, filter_duplicate=filter_duplicate
            ),
            **trainer_kwargs
        )
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.backbone = torch.compile(self.backbone)

        trainer.train()
        if 'push_to_hub' in trainer_kwargs and trainer_kwargs['push_to_hub']:
            trainer.push_to_hub(private=trainer_kwargs.get('hub_private_repo', True))
        self.backbone.save_pretrained(output_dir)

    def evaluate(self, data: Dataset, batch_size: int = 32, threshold: Optional[float] = None, device: Any = None):
        self.backbone.eval()
        data_collator = AngleDataCollator(
            self.tokenizer,
            return_tensors="pt",
            max_length=self.max_length,
            filter_duplicate=False,
        )
        y_trues, y_preds = [], []
        # for X, y in data.make_iter(random=False):
        for features in tqdm(chunked_iter(data, batch_size), desc='Evaluate'):
            X = data_collator(features)
            y = X.pop('labels', None)
            y_trues.extend(y[::2, 0].detach().cpu().numpy())
            with torch.no_grad():
                X.to(device or self.device)
                x_vecs = self.pooler(X).detach().float().cpu().numpy()
            x_vecs = l2_normalize(x_vecs)
            pred = (x_vecs[::2] * x_vecs[1::2]).sum(1)
            y_preds.extend(pred)

        y_trues, y_preds = np.array(y_trues), np.array(y_preds)
        corrcoef = compute_corrcoef(y_trues, y_preds)
        if threshold is None:
            _, accuracy = optimal_threshold(y_trues, y_preds)
        else:
            accuracy = np.mean((y_trues > 0.5) == (y_preds > threshold))
        return corrcoef, accuracy

    def set_prompt(self, prompt: str = Prompts.A):
        self.prompt = prompt
        if self.prompt is not None:
            logger.info('Prompt is set, the prompt will be automatically applied during the encoding phase. '
                        'To disable prompt setting, please configure set_prompt(prompt=None)')

    def encode(self,
               inputs: Union[List[str], Tuple[str], List[Dict], str],
               max_length: Optional[int] = None,
               end_with_eos: bool = False,
               to_numpy: bool = True,
               layer_index: int = -1,
               embedding_size: Optional[int] = None,
               device: Optional[Any] = None):
        """
        encode texts.

        :param inputs: Union[List[str], Tuple[str], List[Dict], str]. Input texts. Required.
        :param max_length: Optional[int]. Default None.
        :param to_numpy: bool. Default True.
        :param layer_index: int. Obtain specific layer's sentence embeddings (for 2DMSE).
        :param embedding_size: Optional[int]. Specify embedding size (for 2DMSE).
        :param device: Optional[Any]. Default None.
        """
        if layer_index != -1 and self.full_backbone is None:
            self.full_backbone = copy.deepcopy(self.backbone)

        if layer_index != -1:
            self.backbone.encoder.layer = self.full_backbone.encoder.layer[:layer_index]

        if device is None:
            device = self.device
        self.backbone.eval()
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        if self.prompt is not None:
            for i, obj in enumerate(inputs):
                assert isinstance(obj, dict), 'The prompt has been set, please pass a dict like {"prompt_key": "text"}'
                inputs[i] = self.prompt.format(**obj)
        max_length = max_length or self.max_length
        if end_with_eos:
            max_length -= 1

        if end_with_eos:
            tok = self.tokenizer(
                inputs,
                padding=False,
                return_attention_mask=False,
                max_length=max_length or self.max_length,
                truncation=True)
            tok['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in tok['input_ids']]
            tok = self.tokenizer.pad(tok, padding=True, return_attention_mask=True, return_tensors='pt')
        else:
            tok = self.tokenizer(
                inputs,
                padding='longest',
                max_length=max_length or self.max_length,
                truncation=True,
                return_tensors='pt')
        tok.to(device)
        with torch.no_grad():
            output = self.pooler(tok, layer_index=layer_index, embedding_size=embedding_size)
        if to_numpy:
            return output.float().detach().cpu().numpy()
        return output
