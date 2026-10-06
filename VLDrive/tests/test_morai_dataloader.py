"""Worker lifecycle/order checks; no CUDA is initialized or required."""
import unittest
import torch
from torch.utils.data import Dataset, SequentialSampler
from data.src.loaders.morai_dataloader import make_morai_loader


class WorkerProbeDataset(Dataset):
    def __len__(self):
        return 9

    def __getitem__(self, index):
        import cv2
        return {'index': index, 'tensor': torch.arange(4) + index,
                'torch_threads': torch.get_num_threads(),
                'cv2_threads': cv2.getNumThreads(),
                'cuda_initialized': torch.cuda.is_initialized()}


class MoraiLoaderTests(unittest.TestCase):
    def test_zero_workers_and_configuration_validation(self):
        data = WorkerProbeDataset()
        loader = make_morai_loader(data, workers=0)
        self.assertIsNone(loader.multiprocessing_context)
        self.assertFalse(loader.persistent_workers)
        self.assertEqual([int(x['index']) for x in loader], list(range(9)))
        for kwargs in ({'workers': -1}, {'batch_size': 0}, {'prefetch_factor': 0},
                       {'workers': 1}, {'sampler': SequentialSampler(data), 'shuffle': True}):
            with self.assertRaises(ValueError):
                make_morai_loader(data, **kwargs)

    def test_spawn_preserves_order_values_and_worker_lifecycle(self):
        data = WorkerProbeDataset()
        generator = torch.Generator().manual_seed(1701)
        loader = make_morai_loader(data, workers=2, generator=generator,
                                  prefetch_factor=2, persistent_workers=True)
        self.assertEqual(loader.multiprocessing_context.get_start_method(), 'spawn')
        first = list(loader)
        worker_pids = [w.pid for w in loader._iterator._workers]
        second = list(loader)
        self.assertEqual(worker_pids, [w.pid for w in loader._iterator._workers])
        for i, (a, b) in enumerate(zip(first, second)):
            self.assertEqual(int(a['index']), i)
            self.assertTrue(torch.equal(a['tensor'], b['tensor']))
            self.assertTrue(torch.equal(a['tensor'][0], torch.arange(4) + i))
            self.assertEqual(int(a['torch_threads']), 1)
            self.assertEqual(int(a['cv2_threads']), 1)
            self.assertFalse(bool(a['cuda_initialized']))
        loader._iterator._shutdown_workers()


if __name__ == '__main__':
    unittest.main()
