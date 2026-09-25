import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useTrainingStore = create(
    persist(
        (set) => ({
            currentJob: null,
            isTraining: false,
            lastStatus: null,
            experimentName: '',
            selectedDataset: '',
            targetColumn: '',
            goalDescription: '',
            configData: {
                epochs: '',
                batch_size: '',
                learning_rate: '',
                n_estimators: '',
                max_depth: '',
                img_size: ''
            },
            imageProblemType: 'image_classification',
            aiPrompt: '',

            setCurrentJob: (job) => set({ currentJob: job }),
            setIsTraining: (isTraining) => set({ isTraining }),
            setLastStatus: (lastStatus) => set({ lastStatus }),
            setExperimentName: (name) => set({ experimentName: name }),
            setSelectedDataset: (id) => set({ selectedDataset: id }),
            setTargetColumn: (col) => set({ targetColumn: col }),
            setGoalDescription: (desc) => set({ goalDescription: desc }),
            setConfigData: (data) => set({ configData: data }),
            setImageProblemType: (type) => set({ imageProblemType: type }),
            setAiPrompt: (prompt) => set({ aiPrompt: prompt }),
            resetTraining: () => set({
                currentJob: null,
                isTraining: false,
                lastStatus: null,
                experimentName: '',
                selectedDataset: '',
                targetColumn: '',
                goalDescription: '',
                configData: {
                    epochs: '',
                    batch_size: '',
                    learning_rate: '',
                    n_estimators: '',
                    max_depth: '',
                    img_size: ''
                },
                imageProblemType: 'image_classification',
                aiPrompt: '',
            }),
        }),
        {
            name: 'inferx-training',
            partialize: (state) => ({
                currentJob: state.currentJob,
                isTraining: state.isTraining,
                lastStatus: state.lastStatus,
                experimentName: state.experimentName,
                selectedDataset: state.selectedDataset,
                targetColumn: state.targetColumn,
                goalDescription: state.goalDescription,
                configData: state.configData,
                imageProblemType: state.imageProblemType,
                aiPrompt: state.aiPrompt,
            })
        }
    )
)
