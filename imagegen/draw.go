// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package imagegen

import (
	"bytes"
	_ "embed"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"math"
	"strings"

	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/goitalic"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

// DrawLabelsOnImage draw text on an image.
func DrawLabelsOnImage(img *image.NRGBA, meme string) {
	f := memeFont
	lines := strings.Split(meme, ",")
	switch len(lines) {
	case 1:
		drawTextOnImage(img, f, 0, lines[0])
	case 2:
		drawTextOnImage(img, f, 0, lines[0])
		drawTextOnImage(img, f, 100, lines[1])
	case 3:
		drawTextOnImage(img, f, 0, lines[0])
		drawTextOnImage(img, f, 50, lines[1])
		drawTextOnImage(img, f, 100, lines[2])
	case 4:
		drawTextOnImage(img, f, 0, lines[0])
		drawTextOnImage(img, f, 30, lines[1])
		drawTextOnImage(img, f, 60, lines[2])
		drawTextOnImage(img, f, 100, lines[3])
	default:
		drawTextOnImage(img, f, 0, lines[0])
		drawTextOnImage(img, f, 20, lines[1])
		drawTextOnImage(img, f, 50, lines[2])
		drawTextOnImage(img, f, 80, lines[3])
		drawTextOnImage(img, f, 100, lines[4])
	}
}

//

var (
	//go:embed mascot.png
	mascotPNG []byte
	//go:embed NotoEmoji-Regular.ttf
	notoEmojiTTF []byte

	mascot        = mustDecodePNG(mascotPNG)
	memeFont      = mustLoadFont(goitalic.TTF)
	notoEmojiFont = mustLoadFont(notoEmojiTTF)
)

func mustDecodePNG(b []byte) *image.NRGBA {
	img, err := decodePNG(b)
	if err != nil {
		panic(err)
	}
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i+3] = img.Pix[i+3] >> 2
	}
	return img
}

func mustLoadFont(b []byte) *opentype.Font {
	f, err := opentype.Parse(b)
	if err != nil {
		panic(err)
	}
	return f
}

// drawTextOnImage draws a single line text on an image.
func drawTextOnImage(img *image.NRGBA, f *opentype.Font, top int, text string) {
	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	d := font.Drawer{Dst: img, Src: image.Black}

	// Do once with a size way too large, then adjust the size.
	// opentype.NewFace() never returns an error.
	face1, _ := opentype.NewFace(memeFont, &opentype.FaceOptions{Size: 1000, DPI: 72})
	face2, _ := opentype.NewFace(notoEmojiFont, &opentype.FaceOptions{Size: 1000, DPI: 72})
	d.Face = &multiface{faces: []font.Face{face1, face2}}

	textWidth := d.MeasureString(text).Round()
	size := 1000. * float64(w) / (250. + float64(textWidth))
	face1, _ = opentype.NewFace(memeFont, &opentype.FaceOptions{Size: size, DPI: 72})
	face2, _ = opentype.NewFace(notoEmojiFont, &opentype.FaceOptions{Size: size, DPI: 72})
	d.Face = &multiface{faces: []font.Face{face1, face2}}
	textWidth = d.MeasureString(text).Round()
	textHeight := d.Face.Metrics().Height.Ceil()
	// The text tends to offshoot on the right so offset it on the left, divide
	// by 4 instead of 2.
	x := (w - textWidth) / 4
	y := top * h / 100
	if y < textHeight {
		y = textHeight
	} else if y > h-40 {
		y = h - 40
	}
	// Draw a crude outline.
	// TODO: It's not super efficient to draw this many (36) times! Make it
	// faster unless it's good enough.
	// TODO: Rasterize at 8x then downsize to reduce aliasing and not have to
	// render so many times.
	radius := 5.
	for i := 0; i < 360; i += 10 {
		a := math.Pi / 180. * float64(i)
		dx := math.Cos(a) * radius
		dy := math.Sin(a) * radius
		dot := fixed.Point26_6{X: fixed.Int26_6((float64(x) + dx) * 64), Y: fixed.Int26_6((float64(y) + dy) * 64)}
		if dot != d.Dot {
			d.Dot = dot
			d.DrawString(text)
		}
	}
	// Draw the final text.
	d.Src = image.White
	d.Dot = fixed.P(x, y)
	d.DrawString(text)
}

// addWatermark adds our mascot onto the image.
func addWatermark(img *image.NRGBA) {
	d := img.Bounds()
	m := mascot.Bounds()
	draw.Draw(img, m.Add(image.Pt(0, d.Dy()-m.Dy())), mascot, image.Point{}, draw.Over)
}

// decodePNG decodes a PNG and ensures it is returned as a NRGBA image.
func decodePNG(b []byte) (*image.NRGBA, error) {
	img, err := png.Decode(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("failed to decode PNG: %w", err)
	}
	switch n := img.(type) {
	case *image.NRGBA:
		return n, nil
	case *image.RGBA:
		// Convert.
		b := n.Bounds()
		dst := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		draw.Draw(dst, dst.Bounds(), n, b.Min, draw.Src)
		return dst, nil
	default:
		return nil, fmt.Errorf("failed to decode PNG: expected NRGBA, got %T", img)
	}
}
