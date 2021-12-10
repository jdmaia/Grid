/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./lib/lattice/Lattice_crc.h

    Copyright (C) 2021

Author: Peter Boyle <paboyle@ph.ed.ac.uk>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */
#pragma once

NAMESPACE_BEGIN(Grid);

template<class vobj> uint32_t crc(Lattice<vobj> & buf)
{
  autoView( buf_v , buf, CpuRead);
  return ::crc32(0L,(unsigned char *)&buf_v[0],(size_t)sizeof(vobj)*buf.oSites());
}

#define CRC(U) std::cout << "FingerPrint "<<__FILE__ <<" "<< __LINE__ <<" "<< #U <<" "<<crc(U)<<std::endl;

NAMESPACE_END(Grid);

