package main

import (
	"io"
	"os"
	"time"

	"github.com/seasonjs/hf-hub/api"
	sd "github.com/seasonjs/stable-diffusion"
)

type stableDiffusion struct {
	m      *sd.Model
	p      sd.FullParams
	prefix string
}

func newStableDiffusion() (*stableDiffusion, error) {
	// Investigate https://github.com/SpenserCai/sd-webui-discord

	hapi, err := api.NewApi()
	if err != nil {
		return nil, err
	}
	// https://github.com/leejet/stable-diffusion.cpp
	modelPath, err := hapi.Model("justinpinkney/miniSD").Get("miniSD.ckpt")
	//modelPath, err := hapi.Model("latent-consistency/lcm-lora-sdv1-5").Get("pytorch_lora_weights.safetensors")
	logger.Info("sd", "model", "miniSD.ckpt", "model_path", modelPath)
	if err != nil {
		return nil, err
	}
	opt := sd.DefaultOptions
	//opt.RngType = sd.STD_DEFAULT_RNG
	model, err := sd.NewAutoModel(opt)
	if err != nil {
		return nil, err
	}
	if err = model.LoadFromFile(modelPath); err != nil {
		model.Close()
		return nil, err
	}
	s := &stableDiffusion{
		m: model,
		p: sd.DefaultFullParams,
		//prefix: "<lora:lcm-lora-sdv1-5:1> ",
	}
	//s.p.CfgScale = 1
	//s.p.SampleMethod = sd.LCM // Incorrect?
	s.p.SampleSteps = 10
	return s, nil
}

func (s *stableDiffusion) Close() error {
	return s.m.Close()
}

func (s *stableDiffusion) genImage(prompt string) (string, error) {
	p := time.Now().Format("2006-01-02T15:04:05") + ".png"
	f, err := os.Create(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	writers := []io.Writer{f}
	// "british short hair cat, high quality"
	if err = s.m.Predict(s.prefix+prompt, s.p, writers); err != nil {
		os.Remove(p)
		return "", err
	}
	return p, nil
}
