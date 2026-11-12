          // the f16 weight widened, then + 0: the loader's table word (it is the astype, then x + 0, which turns -0.0
          // into +0.0 and keeps every other value)
          inter[tn] = static_cast<float>(mat[mat_offset + bn + tn]) + 0.0f;
