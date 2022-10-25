/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/* Portions of this file are additionally subject to the following license
 * and copyright.
 *  MIT License
 *
 *  Copyright (c) 2017-2019 MIT CSAIL, Adobe Systems, and other contributors.
 *
 *  Developed by:
 *
 *    The taco team
 *    http://tensor-compiler.org
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

#pragma once

// arrayEnd is inclusive (i.e. we must be able to access posArray[arrayEnd] safely.
// We'll also generate device code for this if we're going to run on GPUs.
template<typename T>
#if defined(__CUDACC__)
__host__ __device__
#endif
int64_t taco_binarySearchBefore(T posArray, int64_t arrayStart, int64_t arrayEnd, int64_t target) {
  if (posArray[arrayEnd].hi < target) {
    return arrayEnd;
  }
  // The binary search range must be exclusive.
  int64_t lowerBound = arrayStart; // always <= target
  int64_t upperBound = arrayEnd + 1; // always > target
  while (upperBound - lowerBound > 1) {
    // TOOD (rohany): Does this suffer from overflow?
    int64_t mid = (upperBound + lowerBound) / 2;
    auto midRect = posArray[mid];
    // Contains checks lo <= target <= hi.
    if (midRect.contains(target)) { return mid; }
      // Either lo > target or target > hi.
    else if (target > midRect.hi) { lowerBound = mid; }
      // At this point, only lo > target.
    else { upperBound = mid; }
  }
  return lowerBound;
}
