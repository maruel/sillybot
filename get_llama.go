//go:build ignore

package main

import (
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/schollz/progressbar/v3"
)

// Instead, get llamafile + GGUF?
// https://github.com/Mozilla-Ocho/llamafile/releases/download/0.8.8/llamafile-0.8.8

func main() {
	model := "Meta-Llama-3-8B-Instruct.Q5_K_M.llamafile"
	cache, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	dst := filepath.Join(cache, model)
	if _, err := os.Stat(dst); os.IsNotExist(err) {
		url := "https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile/resolve/main/" + model + "?download=true"
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			log.Fatal(err)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			log.Fatal(err)
		}
		defer resp.Body.Close()
		f, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY, 0o755)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		bar := progressbar.DefaultBytes(resp.ContentLength, "downloading")
		if _, err = io.Copy(io.MultiWriter(f, bar), resp.Body); err != nil {
			log.Fatal(err)
		}
	}
}
