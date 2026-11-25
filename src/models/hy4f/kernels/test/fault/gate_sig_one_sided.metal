        // the planted fault (a red check, never served): the one-sided sigmoid 1 / (1 + exp(-y))
        const float sg = 1 / (1 + metal::precise::exp(-yv));
