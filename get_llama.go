//go:build ignore

package main

import (
	"archive/zip"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"

	"github.com/schollz/progressbar/v3"
)

func getGitHubLatestRelease(owner, repo, contentType string) (string, string, error) {
	resp, err := http.Get("https://api.github.com/repos/" + owner + "/" + repo + "/releases")
	if err != nil {
		return "", "", err
	}
	// Just enough of the GitHub API response to be able to parse it.
	data := []struct {
		Assets []struct {
			BrowserDownloadURL string `json:"browser_download_url"`
			ContentType        string `json:"content_type"`
			Name               string `json:"name"`
		} `json:"assets"`
		TagName    string `json:"tag_name"`
		Prerelease bool   `json:"prerelease"`
	}{}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", "", err
	}
	resp.Body.Close()
	for _, l := range data {
		if l.Prerelease {
			continue
		}
		for _, asset := range l.Assets {
			if asset.ContentType == contentType {
				return asset.BrowserDownloadURL, asset.Name, nil
			}
		}
	}
	return "", "", nil
}

func downloadExec(url, dst string) error {
	if _, err := os.Stat(dst); err == nil || !os.IsNotExist(err) {
		return err
	}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	f, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY, 0o755)
	if err != nil {
		return err
	}
	defer f.Close()
	bar := progressbar.DefaultBytes(resp.ContentLength, "downloading")
	_, err = io.Copy(io.MultiWriter(f, bar), resp.Body)
	return err
}

func copyFile(dst, src string) error {
	s, err := os.Open(src)
	if err != nil {
		return err
	}
	defer s.Close()
	st, err := s.Stat()
	if err != nil {
		return err
	}
	d, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY, st.Mode())
	if err != nil {
		return err
	}
	defer d.Close()
	_, err = io.Copy(d, s)
	return err
}

func getHfModelGGUFFromLlamafile(cache, repo, model string) error {
	url := "https://huggingface.co/" + repo + "/resolve/main/" + model + ".llamafile?download=true"
	dst := filepath.Join(cache, model+".llamafile")
	if err := downloadExec(url, dst); err != nil {
		return err
	}
	gguf := model + ".gguf"
	dstgguf := filepath.Join(cache, gguf)
	if _, err := os.Stat(dstgguf); err == nil || !os.IsNotExist(err) {
		return err
	}
	z, err := zip.OpenReader(dst)
	if err != nil {
		return err
	}
	defer z.Close()
	for _, i := range z.File {
		if i.Name == gguf {
			s, err := i.Open()
			if err != nil {
				return err
			}
			defer s.Close()
			d, err := os.OpenFile(dstgguf, os.O_CREATE|os.O_WRONLY, 0o644)
			if err != nil {
				return err
			}
			defer d.Close()
			_, err = io.Copy(d, s)
			return err
		}
	}
	return errors.New("gguf not found")
}

func main() {
	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	cache := filepath.Join(wd, "cache")
	if err = os.MkdirAll(cache, 0o755); err != nil {
		log.Fatal(err)
	}

	execSuffix := ""
	if runtime.GOOS == "windows" {
		execSuffix = ".exe"
	}

	// First, download llamafile from GitHub. We always want the latest and
	// greatest as it is very actively developed and the model we download likely
	// use an older version.
	url, name, err := getGitHubLatestRelease("Mozilla-Ocho", "llamafile", "application/octet-stream")
	if err != nil {
		log.Fatal(err)
	}
	dst := filepath.Join(cache, name+execSuffix)
	if err = downloadExec(url, dst); err != nil {
		log.Fatal(err)
	}
	// Copy it as the default executable to use.
	if err = copyFile(filepath.Join(cache, "llamafile"+execSuffix), dst); err != nil {
		log.Fatal(err)
	}

	// Browse at https://huggingface.co/Mozilla for recent models.
	// https://huggingface.co/Mozilla/Meta-Llama-3-70B-Instruct-llamafile/tree/main
	// is too large for my computers. :(

	// https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile/tree/main
	repo := "Mozilla/Meta-Llama-3-8B-Instruct-llamafile"
	mdl := "Meta-Llama-3-8B-Instruct.Q5_K_M"
	//mdl := "Meta-Llama-3-8B-Instruct.BF16" // Doesn't work on M3 Max.
	//mdl := "Meta-Llama-3-8B-Instruct.F16" // 3x slower than Q5_K_M.

	// https://huggingface.co/jartine/gemma-2-27b-it-llamafile/tree/main
	//repo := "jartine/gemma-2-27b-it-llamafile"
	//mdl := "gemma-2-27b-it.Q6_K"
	// Sync with main.go.
	if err = getHfModelGGUFFromLlamafile(cache, repo, mdl); err != nil {
		log.Fatal(err)
	}
}
