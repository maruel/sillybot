// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package imagegen

import (
	"errors"
	"image"

	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"
)

// multiface leverages multiple fallback fonts to have access to more glyphs.
//
// It's primary use is to add noto emoji as a fallback font.
type multiface struct {
	faces        []font.Face
	r0, r1       rune
	idxr0, idxr1 int
}

func (f *multiface) Close() error {
	var errs []error
	for i := range f.faces {
		errs = append(errs, f.faces[i].Close())
	}
	return errors.Join(errs...)
}

func (f *multiface) Glyph(dot fixed.Point26_6, r rune) (dr image.Rectangle, mask image.Image, maskp image.Point, advance fixed.Int26_6, ok bool) {
	for i := range f.faces {
		if dr, mask, maskp, advance, ok = f.faces[i].Glyph(dot, r); ok {
			break
		}
	}
	return
}

func (f *multiface) GlyphAdvance(r rune) (advance fixed.Int26_6, ok bool) {
	for i := range f.faces {
		if advance, ok = f.faces[i].GlyphAdvance(r); ok {
			f.idxr0 = f.idxr1
			f.r0 = f.r1
			f.idxr1 = i
			f.r1 = r
			break
		}
	}
	return
}

func (f *multiface) GlyphBounds(r rune) (bounds fixed.Rectangle26_6, advance fixed.Int26_6, ok bool) {
	for i := range f.faces {
		if bounds, advance, ok = f.faces[i].GlyphBounds(r); ok {
			break
		}
	}
	return
}

func (f *multiface) Kern(r0, r1 rune) fixed.Int26_6 {
	idxr0 := 0
	if r0 == f.r0 && f.idxr0 >= 0 {
		idxr0 = f.idxr0
	} else if r0 == f.r1 && f.idxr1 >= 0 {
		idxr0 = f.idxr1
	} else {
		idxr0 = f.findFont(r0)
	}

	idxr1 := 0
	if r1 == f.r0 && f.idxr0 >= 0 {
		idxr1 = f.idxr0
	} else if r1 == f.r1 && f.idxr1 >= 0 {
		idxr1 = f.idxr1
	} else {
		idxr1 = f.findFont(r1)
	}

	if idxr0 < 0 || idxr1 < 0 || idxr0 != idxr1 {
		return 0
	}
	return f.faces[idxr0].Kern(r0, r1)
}

func (f *multiface) Metrics() font.Metrics {
	// TODO: We may want to calculate the largest font height?
	return f.faces[0].Metrics()
}

func (f *multiface) findFont(r rune) int {
	for i := range f.faces {
		if _, ok := f.faces[i].GlyphAdvance(r); ok {
			return i
		}
	}
	return -1
}
